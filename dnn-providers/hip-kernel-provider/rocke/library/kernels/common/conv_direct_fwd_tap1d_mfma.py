# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""1-D tap-reuse forward convolution on wave64 MFMA (gfx950).

A dense NHWC forward convolution with a ``1 x S`` filter (any width
dilation and padding, stride 1): every output row is a 1-D convolution of
the matching input row, so a block computes ``BX`` consecutive output pixels
of one row x ``bk`` output channels and reduces over ``ck``-channel chunks
and the S taps:

* **Polyphase lane map.** MFMA column ``c`` (``c < NCOL``, NCOL = the largest
  multiple of the dilation <= 16 * waves along X) owns the ``seg`` output
  pixels ``c % D + D * (seg * (c / D) + u)``. Tap ``s`` of segment pixel
  ``u`` then reads window pixel ``c % D + D * (seg * (c / D) + u + s)``: the
  input fragment ``F_{u+s}``. The ``seg + S - 1`` fragments of a chunk are
  read from LDS once and reused by every tap from registers, for any
  dilation. Columns >= NCOL duplicate column ``c - D`` and store nothing.
* **Window in LDS.** The chunk's input window (``BX + D * (S - 1)`` pixels x
  ``ck`` channels, zero-filled outside the row) is LDS-DMA'd in S pieces
  during the previous chunk (two window buffers; one when C == ck).
* **Weights.** A step is one tap; its ``bk x ck`` weights are LDS-DMA'd one
  step ahead (two stages). 16-byte granules are swizzled by row on the
  global side.

AOT: batch, extents, C and K are kernargs; the filter width, padding,
dilation, tile, chunk width and the single-chunk flag (C == ck) are baked. Limits: KH = 1, stride 1, no height
padding, C % ck == 0, K % bk == 0, packed tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.ir import I32, I64, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    pad_w_reason,
    I32_MAX,
    DirectConvProblem,
    _buf_store_vN,
    _io_type,
    _mfma,
    _trunc_f32,
    emit_direct_params,
    problem_dims_reason,
)

__all__ = ["FwdTap1dMfmaSpec", "build_fwd_tap1d_mfma", "fwd_tap1d_mfma_reason"]

_WAVE = 64
_KT = 2  # 16-channel MFMA tiles per wave along K


@dataclass(frozen=True)
class FwdTap1dMfmaSpec:
    """One configuration of :func:`build_fwd_tap1d_mfma`."""

    problem: DirectConvProblem
    bk: int = 128  # output channels per block tile (32, 64 or 128)
    nwv: int = 8  # waves per block (4 or 8)
    seg: int = 9  # output pixels per MFMA column
    ck: int = 64  # channels per chunk (64, or 32 for C % 64 != 0)
    one: bool = False  # C == ck: a single chunk, one window buffer
    name: str = "fwd_tap1d_mfma"
    wave_size: int = 64

    @property
    def single_chunk(self) -> bool:
        return self.one

    @property
    def nwk(self) -> int:
        return self.bk // (_KT * 16)

    @property
    def nwx(self) -> int:
        return self.nwv // self.nwk

    @property
    def ncol(self) -> int:
        d = self.problem.dil_w
        return d * (self.nwx * 16 // d)

    @property
    def bx(self) -> int:
        return self.ncol * self.seg

    @property
    def win(self) -> int:
        p = self.problem
        return self.bx + p.dil_w * (p.KW - 1)

    def _rows_per_dma(self) -> int:
        return 1024 // (self.ck * 2)

    @property
    def win_dmas(self) -> int:
        return ceil_div(self.win, self._rows_per_dma())

    @property
    def wgt_dmas(self) -> int:
        return ceil_div(self.bk, self._rows_per_dma())

    @property
    def lds_bytes(self) -> int:
        nwb = 1 if self.single_chunk else 2
        return (nwb * self.win_dmas + 2 * self.wgt_dmas) * 1024

    @property
    def threads_per_block(self) -> int:
        return self.nwv * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f1x{p.KW}",
            f"p{p.PAD_W}",
            f"d{p.dil_w}",
            f"k{self.bk}",
            f"w{self.nwv}",
            f"g{self.seg}",
            f"c{self.ck}",
            flags={"one": self.single_chunk, "bf16": p.dtype == "bf16"},
        )

    def grid(self) -> Tuple[int, int, int]:
        p = self.problem
        return (p.N * p.Ho * ceil_div(p.Wo, self.bx) * (p.total_k // self.bk), 1, 1)

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)

    @property
    def workspace_bytes(self) -> int:
        return 0


def fwd_tap1d_mfma_reason(spec: FwdTap1dMfmaSpec, arch: str = "gfx950") -> "str | None":
    p = spec.problem
    target = ArchTarget.from_gfx(arch)
    if target.wave_size != 64 or spec.wave_size != 64:
        return "fwd_tap1d_mfma is a wave64 MFMA kernel"
    if target.async_lds_max_dwords < 4:
        return f"fwd_tap1d_mfma needs the 16-byte LDS DMA (absent on {arch})"
    if p.dtype not in ("fp16", "bf16"):
        return "fp16/bf16 only"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return f"missing mfma_f32_16x16x32_{ab} on {arch}"
    why = problem_dims_reason(p)
    if why is not None:
        return why
    why = pad_w_reason(p)
    if why is not None:
        return why
    if p.groups != 1:
        return "groups must be 1"
    if p.KH != 1 or p.PAD != 0 or p.stride != 1:
        return "1 x S filters, no height padding, stride 1 only"
    if p.PAD_W < 0 or p.PAD_W > p.dil_w * (p.KW - 1):
        return "width padding must be in [0, dilation * (S - 1)]"
    if spec.bk not in (32, 64, 128) or spec.nwv not in (4, 8) or spec.nwv % spec.nwk:
        return "bk must be 32/64/128 and nwv 4/8 with nwv % (bk / 32) == 0"
    if spec.seg < 1 or spec.ncol < 1:
        return "seg must be >= 1 and the dilation at most 16 * waves along X"
    if spec.ck not in (32, 64) or p.total_c % spec.ck:
        return f"ck {spec.ck} must be 32 or 64 and divide C {p.total_c}"
    if spec.one != (p.total_c == spec.ck):
        return f"one (single chunk) must be set exactly when C == ck ({p.total_c} vs {spec.ck})"
    if p.total_k % spec.bk:
        return f"K {p.total_k} must be a multiple of bk {spec.bk}"
    if p.Wo < 1:
        return "empty output"
    if not target.fits_lds(spec.lds_bytes):
        return f"LDS {spec.lds_bytes} B > {target.lds_capacity_bytes}"
    nx = p.N * p.Ho * p.Wo
    for what, nbytes in (
        ("input", 2 * p.N * p.H * p.W * p.total_c),
        ("output", 2 * nx * p.total_k),
        ("weights", 2 * p.total_k * p.KW * p.total_c),
    ):
        if nbytes > I32_MAX:
            return f"{what} exceeds {I32_MAX} B, the i32 buffer range"
    return None


def build_fwd_tap1d_mfma(spec: FwdTap1dMfmaSpec, *, arch: str = "gfx950") -> KernelDef:
    why = fwd_tap1d_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdTap1dMfmaSpec for {arch}: {why}")
    p = spec.problem
    dtype = p.dtype
    io_type = _io_type(dtype)
    S, DIL, PW = p.KW, p.dil_w, p.PAD_W
    BK, NWV, SEG, CK = spec.bk, spec.nwv, spec.seg, spec.ck
    NWK, NCOL, BX, WIN = spec.nwk, spec.ncol, spec.bx, spec.win
    G = CK // 8  # 16-byte granules per LDS row
    RPI = 1024 // (CK * 2)  # LDS rows per DMA instruction
    NH = CK // 32  # 32-channel MFMA halves per chunk
    WIN_I, WGT_I = spec.win_dmas, spec.wgt_dmas
    WIN_P = ceil_div(WIN_I, NWV * S)  # window DMAs per wave and step
    WGT_P = ceil_div(WGT_I, NWV)  # weight DMAs per wave and step
    ONE = spec.single_chunk

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 2
    params = emit_direct_params(b, io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci(I32_MAX)
    p_Wi, p_Wo = params["p_Wi"], params["p_Wo"]
    C, K = params["p_total_c"], params["p_total_k"]

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    kg = b.mod(wave, ci(NWK))
    xg = b.div(wave, ci(NWK))

    def opaque_lane() -> Value:
        # lane-derived addressing is recomputed where it is used instead of
        # being hoisted into long-lived VGPRs
        return b.inline_asm("", "=v,0", [lane], I32, sideeffect=True)

    # block -> (output row, X tile, K tile); K tile fastest
    nkb = b.div(K, ci(BK))
    nxt = b.div(b.add(p_Wo, ci(BX - 1)), ci(BX))
    t = b.block_id_x()
    kb = b.mod(t, nkb)
    t = b.div(t, nkb)
    wo0 = b.mul(b.mod(t, nxt), ci(BX))
    nrow = b.div(t, nxt)  # n * Ho + ho (= input row: KH 1, no height padding)
    nch = b.div(C, ci(CK))

    a_bytes = params["A_bytes"]
    w_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])

    # ---- LDS: one allocation per buffer (static indices keep the alias scopes apart)
    win_s = [
        b.smem_alloc(io_type, [1, WIN_I * 512], name_hint=f"tap1d_x{i}")
        for i in range(1 if ONE else 2)
    ]
    wgt_s = [
        b.smem_alloc(io_type, [1, WGT_I * 512], name_hint=f"tap1d_w{i}")
        for i in range(2)
    ]
    win_a = [b.smem_addr_of(v) for v in win_s]
    wgt_a = [b.smem_addr_of(v) for v in wgt_s]

    def swz_slot(row: Value, gr) -> Value:
        """LDS granule slot of granule ``gr`` in LDS row ``row``."""
        gv = gr if isinstance(gr, Value) else ci(gr)
        return b.xor(gv, b.land(b.xor(row, b.lshr(row, ci(3))), ci(G - 1)))

    def guarded(first: int, last: int, nmax: int, ii: Value, emit):
        """Emit ``emit()`` for DMA instruction ii = first .. last over the waves:
        statically, or behind a wave-uniform ``ii < nmax`` branch."""
        if first >= nmax:
            return
        if last < nmax:
            emit()
            return
        with b.scf_if(b.cmp_lt(ii, ci(nmax))):
            emit()

    def dma_win(ch: Value, buf: int, st: int, live: "Value | None"):
        """Window piece ``st`` of chunk ``ch`` into window buffer ``buf``."""
        rs = b.buffer_rsrc(
            params["A"], a_bytes if live is None else b.select(live, a_bytes, c0)
        )
        for j in range(WIN_P):
            ii = b.add(b.mul(b.add(ci(st * NWV), wave), ci(WIN_P)), ci(j))

            def emit(ii=ii):
                ln = opaque_lane()
                row = b.add(b.mul(ii, ci(RPI)), b.div(ln, ci(G)))
                gr = swz_slot(row, b.mod(ln, ci(G)))  # the swizzle is an involution
                xx = b.add(b.sub(wo0, ci(PW)), row)
                ok = b.land(b.cmp_lt(row, ci(WIN)), b.cmp_ult(xx, p_Wi))
                pix = b.add(b.mul(nrow, p_Wi), xx)
                off = b.mul(
                    b.add(b.add(b.mul(pix, C), b.mul(ch, ci(CK))), b.mul(gr, ci(8))), c2
                )
                dst = b.smem_ptr_add(win_a[buf], b.zext(b.mul(ii, ci(1024)), I64))
                b.async_buffer_load_lds(rs, dst, b.select(ok, off, oob), c0, 4)

            guarded(
                (st * NWV) * WIN_P + j,
                (st * NWV + NWV - 1) * WIN_P + j,
                WIN_I,
                ii,
                emit,
            )

    def dma_wgt(ch: Value, tap: int, stage: int, live: "Value | None"):
        """Weights of (chunk ``ch``, tap) into weight stage ``stage``."""
        rs = (
            w_rsrc
            if live is None
            else b.buffer_rsrc(params["B"], b.select(live, params["B_bytes"], c0))
        )
        soff = b.mul(b.add(b.mul(ci(tap), C), b.mul(ch, ci(CK))), c2)
        for j in range(WGT_P):
            ii = b.add(b.mul(wave, ci(WGT_P)), ci(j))

            def emit(ii=ii):
                ln = opaque_lane()
                row = b.add(b.mul(ii, ci(RPI)), b.div(ln, ci(G)))
                gr = swz_slot(row, b.mod(ln, ci(G)))
                k = b.add(b.mul(kb, ci(BK)), row)
                off = b.mul(b.add(b.mul(b.mul(k, ci(S)), C), b.mul(gr, ci(8))), c2)
                dst = b.smem_ptr_add(wgt_a[stage], b.zext(b.mul(ii, ci(1024)), I64))
                b.async_buffer_load_lds(rs, dst, off, soff, 4)

            guarded(j, (NWV - 1) * WGT_P + j, WGT_I, ii, emit)

    def rd(buf, row: Value, gr: Value) -> Value:
        idx = b.add(b.mul(row, ci(CK)), b.mul(swz_slot(row, gr), ci(8)))
        return b.smem_load_vN(buf, c0, idx, dtype=io_type, n=8)

    nacc = SEG * _KT

    # fragment-read lane bases: computed once (hoisted out of the tap / chunk
    # loops, unlike the DMA lane math) - recomputing them after every barrier
    # puts a VALU chain in front of each tap's first ds_read
    l16r, qr = b.mod(lane, ci(16)), b.div(lane, ci(16))
    colr = b.add(b.mul(xg, ci(16)), l16r)
    if NCOL < NWV // NWK * 16:
        colr = b.select(b.cmp_ge(colr, ci(NCOL)), b.sub(colr, ci(DIL)), colr)
    xrow0 = b.add(b.mod(colr, ci(DIL)), b.mul(b.div(colr, ci(DIL)), ci(DIL * SEG)))
    krow0 = b.add(b.mul(kg, ci(_KT * 16)), l16r)

    def chunk(
        accs: List[Value], ch: Value, wb: int, ws0: int, nxt_live: "Value | None"
    ) -> List[Value]:
        """One chunk read from window buffer wb; the weight stage of tap s is (ws0 + s) & 1.
        nxt_live: there is a next chunk (None: statically none)."""
        accs = list(accs)
        fr: List[List[Value]] = [[None] * (SEG + S - 1) for _ in range(NH)]
        for s in range(S):
            ws = (ws0 + s) & 1
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.s_barrier_bare()
            b.sched_barrier(0)
            if s + 1 < S:
                dma_wgt(ch, s + 1, ws ^ 1, None)
            elif nxt_live is not None:
                dma_wgt(b.add(ch, c1), 0, ws ^ 1, nxt_live)
            if nxt_live is not None:
                dma_win(b.add(ch, c1), wb ^ 1, s, nxt_live)
            a = []
            for h in range(NH):
                gr = b.add(ci(h * 4), qr)
                a.append(
                    [rd(wgt_s[ws], b.add(krow0, ci(kt * 16)), gr) for kt in range(_KT)]
                )
                for v in range(SEG) if s == 0 else (SEG + s - 1,):
                    fr[h][v] = rd(win_s[wb], b.add(xrow0, ci(DIL * v)), gr)
            b.sched_barrier(0)
            for h in range(NH):
                for u in range(SEG):
                    for kt in range(_KT):
                        i = u * _KT + kt
                        accs[i] = _mfma(
                            b, dtype, "16x16x32", a[h][kt], fr[h][u + s], accs[i]
                        )
            b.sched_barrier(0)
        return accs

    zero = b.zero_vec_f32(4)
    vec4 = zero.type
    accs = [zero] * nacc

    # prologue: the whole window of chunk 0 and the weights of its first tap
    for st in range(S):
        dma_win(c0, 0, st, None)
    dma_wgt(c0, 0, 0, None)

    if ONE:
        accs = chunk(accs, c0, 0, 0, None)
    else:
        loop = b.scf_for_iter(
            c0,
            b.div(nch, c2),
            c1,
            [(f"a{i}", v) for i, v in enumerate(accs)],
            iv_name="pair",
            elide_trailing_barrier=False,
        )
        with loop as (pair, la):
            ch = b.mul(pair, c2)
            la = chunk(list(la), ch, 0, 0, b.cmp_lt(b.add(ch, c1), nch))
            la = chunk(la, b.add(ch, c1), 1, S & 1, b.cmp_lt(b.add(ch, c2), nch))
            b.scf_yield(*la)
        accs = list(loop.results)
        tail = b.scf_if_else(b.cmp_eq(b.mod(nch, c2), c1), [vec4] * nacc)
        with tail as (then_ctx, else_ctx):
            with then_ctx:
                b.scf_yield(*chunk(accs, b.sub(nch, c1), 0, 0, None))
            with else_ctx:
                b.scf_yield(*accs)
        accs = list(tail.results)

    # ---- epilogue: acc[u][kt] holds channels k .. k + 3 of pixel px -----------
    d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])
    ln = opaque_lane()
    l16, q = b.mod(ln, ci(16)), b.div(ln, ci(16))
    col = b.add(b.mul(xg, ci(16)), l16)
    colok = b.cmp_lt(col, ci(NCOL))
    px0 = b.add(
        wo0, b.add(b.mod(col, ci(DIL)), b.mul(b.div(col, ci(DIL)), ci(DIL * SEG)))
    )
    kbase = b.add(b.add(b.mul(kb, ci(BK)), b.mul(kg, ci(_KT * 16))), b.mul(q, ci(4)))
    obase = b.mul(nrow, p_Wo)
    for u in range(SEG):
        px = b.add(px0, ci(DIL * u))
        ok = b.land(colok, b.cmp_lt(px, p_Wo))
        for kt in range(_KT):
            k = b.add(kbase, ci(kt * 16))
            off = b.select(ok, b.mul(b.add(b.mul(b.add(obase, px), K), k), c2), oob)
            _buf_store_vN(
                b, dtype, d_rsrc, off, c0, _trunc_f32(b, dtype, accs[u * _KT + kt]), 2
            )
    return b.kernel
