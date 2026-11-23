# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Row-streaming 3x3 forward convolution on wave64 MFMA (gfx950), small K.

A dense NHWC forward convolution, 3x3, stride 1, padding 1, K in {32, 64}.
A block of 4 waves owns ``WT`` output columns x a strip of ``hs`` output rows
x all K channels; waves are split ``K / 16`` along K and ``4 / (K / 16)``
along X (``pw`` output pixels each):

* **Row streaming.** The strip is walked input row by input row. Each input
  row slice (``WT + 2`` pixels x C) is LDS-DMA'd once into a 3-slot ring
  (pixel stride C/8 + 1 granules: conflict-free 16-byte reads) and feeds the
  3 output rows it touches through 3 accumulator slots (ky reuse in
  registers). Rows outside the image are zero-filled by the buffer range
  check; the loop body is branch free (strip-end products land in
  accumulator slots that are zeroed or never stored).
* **Weights in VGPRs.** The 9 taps x C weights of the wave's 16 output
  channels are loaded once per block, so C is baked.
* **Epilogue.** A finished row is staged in LDS (swizzled granules) during
  the next row's MFMAs (4 accumulator slots) and stored a row later with
  full-line 16-byte stores.
* **Hand-counted DMA.** The DMA is issued through inline asm so the
  compiler's wait-count pass does not serialise the ring; the loop waits
  on ``vmcnt`` explicitly (DMA and stores both count).

AOT: batch and extents are kernargs; C, K, ``pw`` and ``hs`` are baked
(C and K must match the problem). Limits: groups 1, packed tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.ir import I32, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    I32_MAX,
    DirectConvProblem,
    _io_type,
    _mfma,
    _trunc_f32,
    emit_direct_params,
    problem_dims_reason,
)

__all__ = ["FwdRowstreamMfmaSpec", "build_fwd_rowstream_mfma", "fwd_rowstream_mfma_reason"]

_WAVE = 64
_NWV = 4  # waves per block
_NSL = 3  # input ring slots
_NA = 4  # accumulator slots (a finished row is staged one row late)
_U = 12  # unroll: lcm(3 staging buffers, 4 accumulator slots, 3 ring slots)
_DMA_ASM = "s_mov_b32 m0, $0\n\ts_nop 0\n\tbuffer_load_dwordx4 $1, $2, 0 offen lds"
_RSRC_W3 = 0x00020000  # buffer descriptor word 3 (gfx950)


@dataclass(frozen=True)
class FwdRowstreamMfmaSpec:
    """One configuration of :func:`build_fwd_rowstream_mfma`."""

    problem: DirectConvProblem
    c: int = 64  # input channels (baked: the weights live in VGPRs)
    k: int = 64  # output channels (32 or 64; one block covers all of them)
    pw: int = 32  # output pixels per wave along X (32 or 64)
    hs: int = 8  # output rows per strip
    name: str = "fwd_rowstream_mfma"
    wave_size: int = 64

    @property
    def nkg(self) -> int:
        return self.k // 16

    @property
    def wt(self) -> int:
        """Output columns per block."""
        return self.pw * (_NWV // self.nkg)

    @property
    def cs(self) -> int:
        """LDS pixel stride in 16-byte granules (odd: conflict-free reads)."""
        return self.c // 8 + 1

    @property
    def dma_per_wave(self) -> int:
        return ceil_div(ceil_div((self.wt + 2) * self.cs, _WAVE), _NWV)

    @property
    def slot_bytes(self) -> int:
        return self.dma_per_wave * _NWV * 1024

    @property
    def stage_bytes(self) -> int:
        return self.wt * self.k * 2

    @property
    def stores_per_thread(self) -> int:
        return self.wt * (self.k // 8) // (_NWV * _WAVE)

    @property
    def lds_bytes(self) -> int:
        return _NSL * self.slot_bytes + 3 * self.stage_bytes

    @property
    def threads_per_block(self) -> int:
        return _NWV * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"c{self.c}",
            f"k{self.k}",
            f"w{self.pw}",
            f"h{self.hs}",
            flags={"bf16": self.problem.dtype == "bf16"},
        )

    def grid(self) -> Tuple[int, int, int]:
        p = self.problem
        return (p.N * ceil_div(p.Ho, self.hs) * ceil_div(p.Wo, self.wt), 1, 1)

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)

    @property
    def workspace_bytes(self) -> int:
        return 0


def fwd_rowstream_mfma_reason(spec: FwdRowstreamMfmaSpec, arch: str = "gfx950") -> "str | None":
    p = spec.problem
    target = ArchTarget.from_gfx(arch)
    if target.wave_size != 64 or spec.wave_size != 64:
        return "fwd_rowstream_mfma is a wave64 MFMA kernel"
    if target.async_lds_max_dwords < 4:
        return f"fwd_rowstream_mfma needs the 16-byte LDS DMA (absent on {arch})"
    if p.dtype not in ("fp16", "bf16"):
        return "fp16/bf16 only"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32):
        return f"missing mfma_f32_16x16x32_{ab} on {arch}"
    why = problem_dims_reason(p)
    if why is not None:
        return why
    if p.groups != 1:
        return "groups must be 1"
    if (p.KH, p.KW, p.PAD, p.PAD_W, p.stride, p.dil_h, p.dil_w) != (3, 3, 1, 1, 1, 1, 1):
        return "3x3, padding 1, stride 1, dilation 1 only"
    if spec.k not in (32, 64) or p.total_k != spec.k:
        return f"k must be 32 or 64 and equal K ({spec.k} vs {p.total_k})"
    if spec.c % 32 or not 32 <= spec.c <= 192 or p.total_c != spec.c:
        return f"c must be a multiple of 32 in [32, 192] and equal C ({spec.c} vs {p.total_c})"
    if spec.pw not in (32, 64):
        return "pw must be 32 or 64"
    if spec.hs < 2:
        return "hs must be >= 2"
    if not target.fits_lds(spec.lds_bytes):
        return f"LDS {spec.lds_bytes} B > {target.lds_capacity_bytes}"
    for what, nbytes in (
        ("input", 2 * p.N * p.H * p.W * p.total_c),
        ("output", 2 * p.N * p.Ho * p.Wo * p.total_k),
        ("weights", 2 * p.total_k * 9 * p.total_c),
    ):
        if nbytes > I32_MAX:
            return f"{what} exceeds {I32_MAX} B, the i32 buffer range"
    return None


def build_fwd_rowstream_mfma(spec: FwdRowstreamMfmaSpec, *, arch: str = "gfx950") -> KernelDef:
    why = fwd_rowstream_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdRowstreamMfmaSpec for {arch}: {why}")
    p = spec.problem
    dtype = p.dtype
    io_type = _io_type(dtype)
    C, K, PW, HS = spec.c, spec.k, spec.pw, spec.hs
    NKG, WT, CS = spec.nkg, spec.wt, spec.cs
    NPX = WT + 2  # input pixels per LDS row
    C8 = C // 8
    NIW = spec.dma_per_wave
    NC32 = C // 32
    NP = PW // 16
    G = K // 8  # 16-byte output granules per pixel
    NST = spec.stores_per_thread
    NIN = HS + 2  # input rows per strip

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 1
    params = emit_direct_params(b, io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0 = ci(0)
    oob = ci(I32_MAX)
    H, W = params["p_Hi"], params["p_Wi"]

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    kg = b.mod(wave, ci(NKG))
    pg = b.div(wave, ci(NKG))

    # block -> (image, strip, X tile); strips adjacent on one XCD share halo rows in L2
    nst = b.div(b.add(H, ci(HS - 1)), ci(HS))
    nwt = b.div(b.add(W, ci(WT - 1)), ci(WT))
    nb = b.mul(b.mul(params["p_N"], nst), nwt)
    bid = b.block_id_x()
    q8 = b.div(nb, ci(8))
    grouped = b.add(b.mul(b.mod(bid, ci(8)), q8), b.div(bid, ci(8)))
    blk = b.select(b.cmp_eq(b.mod(nb, ci(8)), c0), grouped, bid)
    h0 = b.mul(b.mod(blk, nst), ci(HS))
    t = b.div(blk, nst)
    w0 = b.mul(b.mod(t, nwt), ci(WT))
    n = b.div(t, nwt)

    # ---- LDS: one allocation per ring slot / staging buffer (static indices)
    slot_s = [b.smem_alloc(io_type, [1, spec.slot_bytes // 2], name_hint=f"rs_x{i}") for i in range(_NSL)]
    stg_s = [b.smem_alloc(io_type, [1, spec.stage_bytes // 2], name_hint=f"rs_y{i}") for i in range(3)]
    slot_m0 = [
        b.to_sgpr_u32(b.add(b.trunc(b.smem_addr_of(s), I32), b.mul(wave, ci(1024)))) for s in slot_s
    ]

    # input buffer descriptor as 4 scalar words (the DMA asm takes an SGPR quad)
    a_addr = b.global_ptr_to_i64(params["A"])
    a_lo = b.to_sgpr_u32(b.trunc(a_addr, I32))
    a_hi = b.to_sgpr_u32(b.land(b.trunc(b.lshr(a_addr, b.const_i64(32)), I32), ci(0xFFFF)))
    a_rsrc = b.vec_pack([a_lo, a_hi, params["A_bytes"], ci(_RSRC_W3)], I32)
    rowpix = b.mul(W, ci(C * 2))  # bytes per input row

    # per-lane DMA source offsets within a row: chunk (j * 4 + wave) * 64 + lane -> (px, c8)
    vo: List[Value] = []
    for j in range(NIW):
        qq = b.add(b.mul(b.add(ci(j * _NWV), wave), ci(_WAVE)), lane)
        px = b.div(qq, ci(CS))
        c8 = b.sub(qq, b.mul(px, ci(CS)))
        wc = b.add(b.sub(w0, ci(1)), px)
        ok = b.land(b.land(b.cmp_lt(px, ci(NPX)), b.cmp_lt(c8, ci(C8))), b.cmp_ult(wc, W))
        vo.append(b.select(ok, b.mul(b.add(b.mul(wc, ci(C)), b.mul(c8, ci(8))), ci(2)), oob))

    def dma_row(i: Value, sl: int, want: "Value | None") -> None:
        """Input rel row ``i`` (absolute h0 - 1 + i; outside the image or
        unwanted -> zero fill) into ring slot ``sl``."""
        r = b.add(b.sub(h0, ci(1)), i)
        ok = b.cmp_ult(r, H)
        if want is not None:
            ok = b.land(ok, want)
        rb = b.select(ok, b.mul(b.add(b.mul(n, H), r), rowpix), ci(-(2**31)))
        rb = b.to_sgpr_u32(rb)
        for j in range(NIW):
            m0 = slot_m0[sl] if j == 0 else b.add(slot_m0[sl], ci(j * _NWV * 1024))
            b.inline_asm(_DMA_ASM, "s,v,s,~{m0},~{memory}", [m0, b.add(rb, vo[j]), a_rsrc])

    # ---- weights -> VGPRs: A fragment (ky, kx, c): lane = (k = kg*16 + lane%16, c8 = c*4 + lane/16)
    w_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])
    l16, l4 = b.mod(lane, ci(16)), b.div(lane, ci(16))
    wbase = b.mul(b.add(b.mul(b.add(b.mul(kg, ci(16)), l16), ci(9 * C)), b.mul(l4, ci(8))), ci(2))
    wr = [
        [
            [b.buffer_load_vN(w_rsrc, wbase, ci(((ky * 3 + kx) * C + c * 32) * 2), io_type, 8) for c in range(NC32)]
            for kx in range(3)
        ]
        for ky in range(3)
    ]

    d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])
    zero = b.zero_vec_f32(4)
    pbase = b.mul(pg, ci(PW))
    # fragment-read lane base (elements), hoisted out of the row loop
    rbase = b.mul(b.add(b.mul(b.add(pbase, l16), ci(CS)), l4), ci(8))
    # staging write lane terms: channel k = kg*16 + (lane/16)*4 -> granule g, half
    kst = b.add(b.mul(kg, ci(16)), b.mul(l4, ci(4)))
    gst, hst = b.lshr(kst, ci(3)), b.land(b.lshr(kst, ci(2)), ci(1))

    def stage_write(acc, s: int, sb: int):
        """Accumulator slot s -> staging buffer sb (pixel px, granule g ^ (px % G)); zero the slot."""
        for pp in range(NP):
            pxv = b.add(b.add(pbase, ci(pp * 16)), l16)
            gs = b.xor(gst, b.land(pxv, ci(G - 1)))
            idx = b.add(b.mul(b.add(b.mul(pxv, ci(G)), gs), ci(8)), b.mul(hst, ci(4)))
            b.smem_store_vN(stg_s[sb], [c0, idx], _trunc_f32(b, dtype, acc[s][pp]), 4)
            acc[s][pp] = zero

    n_oob = [0]

    def store_oob() -> Value:
        """A fresh out-of-range store offset (>= 2^31 > D_bytes). Every masked store gets its
        own: two stores the compiler can prove hit one address would be merged, which breaks
        the hand-counted vmcnt (stores count)."""
        n_oob[0] += 1
        return ci(-(2**31) + 16 * n_oob[0])

    def stage_store(sb: int, j: Value, dummy: bool) -> None:
        """Staging buffer sb -> output rel row j (full row slice, 16 B per thread); dummy: OOB only."""
        if not dummy:
            o = b.add(h0, j)
            rok = b.land(b.land(b.cmp_ge(j, c0), b.cmp_lt(j, ci(HS))), b.cmp_lt(o, H))
            rowe = b.mul(b.add(b.mul(n, H), o), W)
        for u in range(NST):
            qq = b.add(ci(u * _NWV * _WAVE), tid)
            pxv, g = b.div(qq, ci(G)), b.mod(qq, ci(G))
            if dummy:
                v = b.zero_vec(io_type, 8)
                off = store_oob()
            else:
                gs = b.xor(g, b.land(pxv, ci(G - 1)))
                v = b.smem_load_vN(stg_s[sb], c0, b.mul(b.add(b.mul(pxv, ci(G)), gs), ci(8)), dtype=io_type, n=8)
                col = b.add(w0, pxv)
                ok = b.land(rok, b.cmp_lt(col, W))
                off = b.select(ok, b.mul(b.add(b.mul(b.add(rowe, col), ci(K)), b.mul(g, ci(8))), ci(2)), store_oob())
            b.buffer_store_vN(d_rsrc, off, c0, v)

    def compute_row(acc, ia: int, sl: int, ws: int, wsb: int) -> None:
        """All 3 ky of input row slot sl into accumulator slots (ia - ky) % NA; steps (c, p) with
        the next step's 3 B fragments read ahead; the deferred row's slot ws is staged after step 0."""
        ns = NC32 * NP

        def rd(st: int) -> List[Value]:
            c, pp = st // NP, st % NP
            return [
                b.smem_load_vN(slot_s[sl], c0, b.add(rbase, ci(((pp * 16 + kx) * CS + c * 4) * 8)), dtype=io_type, n=8)
                for kx in range(3)
            ]

        nxt = rd(0)
        for st in range(ns):
            c, pp = st // NP, st % NP
            cur = nxt
            if st + 1 < ns:
                nxt = rd(st + 1)
            b.sched_barrier(0)
            for kx in range(3):
                for ky in range(3):
                    s = (ia - ky) % _NA
                    acc[s][pp] = _mfma(b, dtype, "16x16x32", wr[ky][kx][c], cur[kx], acc[s][pp])
            if st == 0:
                stage_write(acc, ws, wsb)
            b.sched_barrier(0)

    def sync(vmcnt: int = -1) -> None:
        """Wait (vmcnt by hand, all LDS ops), barrier, then a compiler memory fence: the
        DMA is invisible to the compiler, so no LDS read may be hoisted above the wait."""
        b.s_waitcnt(vmcnt=vmcnt, lgkmcnt=0)
        b.s_barrier_bare()
        b.inline_asm("", "~{memory}", [])
        b.sched_barrier(0)

    def iteration(acc, i: Value, u: int, i_static: "int | None") -> None:
        """Input rel row i; u = i mod U (static ring / slot / staging indices)."""
        sync(NIW * (_NSL - 2) + NST * (_NSL - 1))  # row i landed (+ the stores issued after it)
        nr = b.add(i, ci(_NSL - 1))
        if i_static is None:
            want = b.cmp_lt(nr, ci(NIN))
        else:
            want = None
            if i_static + _NSL - 1 >= NIN:
                want = b.cmp_lt(ci(0), ci(0))  # constant false: zero fill (fixed vmcnt bookkeeping)
        dma_row(nr, (u + _NSL - 1) % _NSL, want)
        stage_store((u + 2) % 3, b.sub(i, ci(4)), False)  # row i - 4, staged last iteration
        compute_row(acc, u % _NA, u % _NSL, (u + _NA - 3) % _NA, u % 3)

    acc = [[zero] * NP for _ in range(_NA)]
    # prologue: rows 0 .. NSL-2, each followed by a dummy store group (fixed vmcnt bookkeeping)
    for i in range(_NSL - 1):
        dma_row(ci(i), i, None)
        stage_store(0, c0, True)

    nfull = NIN // _U if NIN > _U else 0
    if nfull:
        flat = [v for row in acc for v in row]
        loop = b.scf_for_iter(
            c0, ci(nfull), ci(1), [(f"a{x}", v) for x, v in enumerate(flat)], iv_name="grp",
            elide_trailing_barrier=False,
        )
        with loop as (grp, la):
            la = list(la)
            ac = [la[s * NP:(s + 1) * NP] for s in range(_NA)]
            i0 = b.mul(grp, ci(_U))
            for u in range(_U):
                iteration(ac, b.add(i0, ci(u)), u, None)
            b.scf_yield(*[v for row in ac for v in row])
        res = list(loop.results)
        acc = [res[s * NP:(s + 1) * NP] for s in range(_NA)]
    for i in range(nfull * _U, NIN):
        iteration(acc, ci(i), i % _U, i)

    # tail: row HS-2 (staged in the last iteration), then row HS-1 (still in its slot).
    # Drain the (zero-fill) DMAs of the last iterations too: an LDS DMA still in flight
    # when the block retires would land in the next block's LDS.
    sync(NST)
    stage_store((NIN - 1) % 3, ci(HS - 2), False)
    stage_write(acc, (HS - 1) % _NA, NIN % 3)
    sync()
    stage_store(NIN % 3, ci(HS - 1), False)
    return b.kernel
