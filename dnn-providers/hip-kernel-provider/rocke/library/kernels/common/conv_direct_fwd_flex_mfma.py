# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Flattened-pixel implicit-GEMM forward convolution on wave64 MFMA (gfx950).

A dense NHWC forward convolution with any filter, stride, dilation and
padding, as a GEMM of output channels (MFMA A = weights) by flattened output
pixels ``X = N * Ho * Wo`` (MFMA B = input), reducing over taps x 64-channel
chunks:

* **Tile.** A block computes X256 x K``bk`` with 8 waves in two ping-pong
  groups (group g owns pixel half g); a wave owns 64 pixels x ``16 * kt``
  channels (``kt`` = 8 for ``bk`` 256, 4 for ``bk`` 128). One *step* is one
  tap and one 64-channel chunk: per 32-channel half the wave reads its 4 input
  and ``kt`` weight fragments from LDS and runs ``4 * kt`` ``mfma_16x16x32``.
* **LDS-DMA for both operands.** Two stages of (input X256 x 64 channels,
  weights ``bk`` x 64 channels), 16-byte ``buffer_load ... lds`` with the
  granule swizzle applied on the global side. im2col is a per-lane pixel base
  (tile constant) plus a scalar tap offset; taps outside the image are
  dropped by an out-of-range offset (zero fill).
* **Ping-pong DMA schedule.** Barrier instances alternate between the
  groups' memory and MFMA phases; the DMA of step ``j + 1`` is issued right
  after instance ``2j`` and waited right before ``2j + 2`` (group 0 issues at
  its memory-phase start and waits at its MFMA-phase end, group 1, one
  barrier behind, issues at its MFMA-phase start and waits at its
  memory-phase end), so every DMA has a whole step to land.
* **Tap skipping.** A tile's output rows (and columns, when it lies inside
  one output row) bound the rows / columns each tap reads; the taps whose
  window is entirely padding form a prefix / suffix of the filter rows and
  columns and are skipped (atrous layers skip up to a third of the steps).
* **Persistent grid** of up to ``PERSIST`` blocks; block b takes item
  ``i * G + (b + i * ROT) % G`` in round i (the rotation spreads the tile
  classes of tap skipping over the blocks); the step stream, and so the DMA
  look-ahead, runs across items.
* **Split-C.** With ``split_c`` > 1 an item is (tile, piece of the chunks)
  and the block writes an fp32 slab of the packed ``[split_c, N, Ho, Wo, K]``
  workspace that ``direct_split_reduce`` folds into the output.
* **Register epilogue.** The output-channel order inside each 16-channel MFMA
  tile is interleaved by the weight DMA so that one ``permlane32_swap`` per
  dword pair gives every lane 8 consecutive channels: 16-byte stores.

AOT: batch, extents, C and K are kernargs; the filter, padding, stride,
dilation, tile and split are baked. Limits: C % 64 == 0, K % 8 == 0, at most
32 taps, packed tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.ir import BF16, F16, I32, I64, IRBuilder, KernelDef, Value
from rocke.core.ir import VectorType
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    I32_MAX,
    DirectConvProblem,
    _buf_store_vN,
    _io_type,
    _mfma,
    _trunc_f32,
    emit_direct_params,
    pad_w_reason,
    problem_dims_reason,
)

__all__ = ["FwdFlexMfmaSpec", "build_fwd_flex_mfma", "fwd_flex_mfma_reason"]

_WAVE = 64
_CK = 64  # channels per chunk
_BX = 256  # pixels per block tile
_XT = 4  # 16-pixel MFMA tiles per wave
PERSIST = 256  # blocks of the persistent grid (one per gfx950 CU)
ROT = 97  # per-round rotation of the item walk


@dataclass(frozen=True)
class FwdFlexMfmaSpec:
    """One configuration of :func:`build_fwd_flex_mfma`."""

    problem: DirectConvProblem
    bk: int = 256  # output channels per block tile (256 or 128)
    split_c: int = 1  # pieces of the channel chunks (fp32 slabs when > 1)
    name: str = "fwd_flex_mfma"
    wave_size: int = 64

    @property
    def kt(self) -> int:
        return 8 if self.bk == 256 else 4

    @property
    def threads_per_block(self) -> int:
        return 8 * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}x{p.PAD_W}",
            f"s{p.stride}",
            f"d{p.dil_h}x{p.dil_w}",
            f"k{self.bk}",
            f"sc{self.split_c}",
            flags={"bf16": p.dtype == "bf16"},
        )

    def items(self) -> int:
        p = self.problem
        tiles = ceil_div(p.N * p.Ho * p.Wo, _BX) * ceil_div(p.total_k, self.bk)
        return tiles * self.split_c

    def grid(self) -> Tuple[int, int, int]:
        return (min(self.items(), PERSIST), 1, 1)

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)

    @property
    def workspace_bytes(self) -> int:
        p = self.problem
        if self.split_c == 1:
            return 0
        return 4 * self.split_c * p.N * p.Ho * p.Wo * p.total_k


def fwd_flex_mfma_reason(spec: FwdFlexMfmaSpec, arch: str = "gfx950") -> "str | None":
    p = spec.problem
    target = ArchTarget.from_gfx(arch)
    if target.wave_size != 64 or spec.wave_size != 64:
        return "fwd_flex_mfma is a wave64 MFMA kernel"
    if target.async_lds_max_dwords < 4:
        return f"fwd_flex_mfma needs the 16-byte LDS DMA (absent on {arch})"
    stage = (_BX + spec.bk) * _CK * 2
    if not target.fits_lds(2 * stage):
        return f"LDS {2 * stage} B > {target.lds_capacity_bytes}"
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
    if p.groups != 1:
        return "groups must be 1"
    if p.stride < 1 or p.dil_h < 1 or p.dil_w < 1:
        return (
            f"stride and dilation must be >= 1 "
            f"(got s{p.stride} d{p.dil_h}x{p.dil_w})"
        )
    why = pad_w_reason(p)
    if why is not None:
        return why
    if spec.bk not in (128, 256):
        return "bk must be 128 or 256"
    if p.total_c % _CK:
        return f"C {p.total_c} must be a multiple of {_CK}"
    if p.total_k % 8:
        return f"K {p.total_k} must be a multiple of 8 (16-byte output writes)"
    if spec.split_c < 1 or spec.split_c > p.total_c // _CK:
        return f"split_c {spec.split_c} must be in [1, C / {_CK}]"
    if p.KH * p.KW > 32:
        return "more than 32 taps (one valid-tap bit per tap)"
    if p.Ho < 1 or p.Wo < 1:
        return "empty output"
    # every buffer offset is an i32 over the whole tensor (or workspace slab)
    nx = p.N * p.Ho * p.Wo
    for what, nbytes in (
        ("input", 2 * p.N * p.H * p.W * p.total_c),
        ("output", 2 * nx * p.total_k),
        ("weights", 2 * p.total_k * p.KH * p.KW * p.total_c),
        ("workspace slab", 4 * nx * p.total_k if spec.split_c > 1 else 0),
    ):
        if nbytes > I32_MAX:
            return f"{what} exceeds {I32_MAX} B, the i32 buffer range"
    # the whole workspace's size is an i32 kernarg of the conv and the fold
    if spec.workspace_bytes > I32_MAX:
        return f"workspace exceeds {I32_MAX} B, the i32 byte-count kernarg"
    return None


def build_fwd_flex_mfma(spec: FwdFlexMfmaSpec, *, arch: str = "gfx950") -> KernelDef:
    why = fwd_flex_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdFlexMfmaSpec for {arch}: {why}")
    p = spec.problem
    dtype = p.dtype
    io_type = _io_type(dtype)
    BK, KT, SP = spec.bk, spec.kt, spec.split_c
    WKN = BK // (KT * 16)  # waves of a group along K
    WXN = 4 // WKN  # ... and along X
    assert WKN * WXN == 4 and WXN * 64 * 2 == _BX
    NXI, NWI = _BX // 64, BK // 64  # DMA instructions per wave per step
    R, S = p.KH, p.KW
    SH, PH, PW, DH, DW = p.stride, p.PAD, p.PAD_W, p.dil_h, p.dil_w
    IN_E, W_E = _BX * _CK, BK * _CK  # elements per stage

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 2
    params = emit_direct_params(b, io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci(I32_MAX)
    p_N, p_Hi, p_Wi = params["p_N"], params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C, K = params["p_total_c"], params["p_total_k"]

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    g = b.div(wave, ci(4))
    wq = b.mod(wave, ci(4))
    wk = b.mod(wq, ci(WKN))
    wx = b.div(wq, ci(WKN))

    def opaque_lane() -> Value:
        # lane-derived addressing is recomputed where it is used instead of
        # being hoisted into long-lived VGPRs
        return b.inline_asm("", "=v,0", [lane], I32, sideeffect=True)

    hw = b.mul(p_Ho, p_Wo)
    NX = b.mul(p_N, hw)
    nxt = b.div(b.add(NX, ci(_BX - 1)), ci(_BX))
    nkt = b.div(b.add(K, ci(BK - 1)), ci(BK))
    TI = b.mul(b.mul(nxt, nkt), ci(SP))
    G = b.smin(TI, ci(PERSIST))
    nch = b.div(C, ci(_CK))
    RSC = b.mul(ci(R * S), C)
    blk = b.block_id_x()
    full, part = b.div(TI, G), b.mod(TI, G)
    last_ok = b.cmp_lt(b.mod(b.add(blk, b.mul(full, ci(ROT))), G), part)
    rounds = b.add(full, b.select(last_ok, c1, c0))

    def tile_of(i: Value) -> Value:
        return b.add(b.mul(i, G), b.mod(b.add(blk, b.mul(i, ci(ROT))), G))

    # ---- item -> (tile, piece, valid tap ranges) -----------------------------
    def item_info(t: Value) -> Dict[str, Value]:
        tl = b.div(t, ci(SP))
        pc = b.mod(t, ci(SP))
        tx = b.div(tl, nkt)
        tk = b.mod(tl, nkt)
        x0 = b.mul(tx, ci(_BX))
        x1 = b.sub(b.smin(b.add(x0, ci(_BX)), NX), c1)
        n0, n1 = b.div(x0, hw), b.div(x1, hw)
        same = b.cmp_eq(n0, n1)
        q0 = b.sub(x0, b.mul(n0, hw))
        q1 = b.sub(x1, b.mul(n0, hw))
        ho0 = b.select(same, b.div(q0, p_Wo), c0)
        ho1 = b.select(same, b.div(q1, p_Wo), b.sub(p_Ho, c1))
        row = b.land(same, b.cmp_eq(ho0, ho1))
        wo0 = b.select(row, b.sub(q0, b.mul(ho0, p_Wo)), c0)
        wo1 = b.select(row, b.sub(q1, b.mul(ho1, p_Wo)), b.sub(p_Wo, c1))
        # valid taps are an interval: a prefix of the rows reads only the top
        # padding (lo < 0 at the tile's last row) and a suffix only the bottom
        lo_h = b.sub(b.mul(ho1, ci(SH)), ci(PH))
        hi_h = b.sub(b.mul(ho0, ci(SH)), ci(PH))
        lo_w = b.sub(b.mul(wo1, ci(SH)), ci(PW))
        hi_w = b.sub(b.mul(wo0, ci(SH)), ci(PW))
        r0, r1, s0, s1 = c0, ci(R), c0, ci(S)
        for r in range(R):
            r0 = b.add(r0, b.select(b.cmp_lt(b.add(lo_h, ci(r * DH)), c0), c1, c0))
            r1 = b.sub(r1, b.select(b.cmp_ge(b.add(hi_h, ci(r * DH)), p_Hi), c1, c0))
        for s in range(S):
            s0 = b.add(s0, b.select(b.cmp_lt(b.add(lo_w, ci(s * DW)), c0), c1, c0))
            s1 = b.sub(s1, b.select(b.cmp_ge(b.add(hi_w, ci(s * DW)), p_Wi), c1, c0))
        # a tile reading only padding still runs one (zero) step
        empty = b.lor(b.cmp_ge(r0, r1), b.cmp_ge(s0, s1))
        r0, r1 = b.select(empty, c0, r0), b.select(empty, c1, r1)
        s0, s1 = b.select(empty, c0, s0), b.select(empty, c1, s1)
        ch0 = b.div(b.mul(pc, nch), ci(SP))
        ch1 = b.div(b.mul(b.add(pc, c1), nch), ci(SP))
        return dict(pc=pc, tx=tx, tk=tk, r0=r0, r1=r1, s0=s0, s1=s1, ch0=ch0, ch1=ch1)

    # ---- DMA cursor ------------------------------------------------------------
    SC = ["i", "t", "ch", "ch1", "r", "s", "r0", "r1", "s0", "s1"]
    LN = [f"xb{i}" for i in range(NXI)] + [f"vm{i}" for i in range(NXI)] + ["wo"]

    def lanes_of(info) -> Dict[str, Value]:
        ln = opaque_lane()
        l8 = b.div(ln, ci(8))
        c8 = b.xor(b.mod(ln, ci(8)), l8)
        out = {}
        for i in range(NXI):
            x = b.add(
                b.mul(info["tx"], ci(_BX)),
                b.add(b.mul(b.add(b.mul(wave, ci(NXI)), ci(i)), ci(8)), l8),
            )
            n = b.div(x, hw)
            rem = b.sub(x, b.mul(n, hw))
            ho = b.div(rem, p_Wo)
            wo = b.sub(rem, b.mul(ho, p_Wo))
            hb = b.sub(b.mul(ho, ci(SH)), ci(PH))
            wb = b.sub(b.mul(wo, ci(SH)), ci(PW))
            pix = b.add(b.mul(b.add(b.mul(n, p_Hi), hb), p_Wi), wb)
            out[f"xb{i}"] = b.mul(b.add(b.mul(pix, C), b.mul(c8, ci(8))), c2)
            # bit (r * S + s) set: tap (r, s) reads an in-image pixel
            vm = c0
            xok = b.cmp_lt(x, NX)
            for r in range(R):
                hok = b.land(xok, b.cmp_ult(b.add(hb, ci(r * DH)), p_Hi))
                for s_ in range(S):
                    ok = b.land(hok, b.cmp_ult(b.add(wb, ci(s_ * DW)), p_Wi))
                    vm = b.add(
                        vm,
                        b.select(
                            ok,
                            (
                                ci(1 << (r * S + s_))
                                if r * S + s_ < 31
                                else ci(-(1 << 31))
                            ),
                            c0,
                        ),
                    )
            out[f"vm{i}"] = vm
        # weight rows: wave w DMAs LDS rows (w * NWI + i) * 8 + l / 8. Row
        # 4g + j of a 16-row tile holds channel (g & 1) * 8 + (g >> 1) * 4 + j,
        # so the permlane32_swap pairs of the epilogue (lane rows 0/2 and 1/3)
        # hold 8 consecutive channels. Row + 8 is channel + 4 and row + 16
        # channel + 16, so the lane part is one offset and instruction i adds
        # a scalar.
        # Tail rows (k >= K) fall past B_bytes: zero fill, never stored.
        m = l8
        gq = b.div(m, ci(4))
        chan = b.add(b.mul(gq, ci(8)), b.mod(m, ci(4)))
        k = b.add(b.add(b.mul(info["tk"], ci(BK)), b.mul(wave, ci(NWI * 8))), chan)
        out["wo"] = b.mul(b.add(b.mul(k, RSC), b.mul(c8, ci(8))), c2)
        return out

    def cursor_at(i: Value) -> Dict[str, Value]:
        t = tile_of(i)
        info = item_info(t)
        cur = dict(
            i=i, t=t, ch=info["ch0"], ch1=info["ch1"], r=info["r0"], s=info["s0"]
        )
        cur.update({k: info[k] for k in ("r0", "r1", "s0", "s1")})
        cur.update(lanes_of(info))
        return cur

    def cur_adv(cur) -> Dict[str, Value]:
        s = b.add(cur["s"], c1)
        ws = b.cmp_ge(s, cur["s1"])
        s = b.select(ws, cur["s0"], s)
        r = b.add(cur["r"], b.select(ws, c1, c0))
        wr = b.cmp_ge(r, cur["r1"])
        r = b.select(wr, cur["r0"], r)
        ch = b.add(cur["ch"], b.select(wr, c1, c0))
        wc = b.cmp_ge(ch, cur["ch1"])
        keep = dict(cur, s=s, r=r, ch=ch)
        names = SC + LN
        branch = b.scf_if_else(wc, [I32] * len(names))
        with branch as (then_ctx, else_ctx):
            with then_ctx:
                nxt = cursor_at(b.add(cur["i"], c1))
                b.scf_yield(*[nxt[k] for k in names])
            with else_ctx:
                b.scf_yield(*[keep[k] for k in names])
        return branch.results

    # ---- LDS --------------------------------------------------------------------
    # one allocation per stage and operand: with static stage indices the
    # lowering's per-allocation alias scopes let the waitcnt pass tell a DMA
    # into one stage from the reads of the other (a runtime stage index made
    # it wait for every in-flight DMA before the fragment reads)
    in_s = [b.smem_alloc(io_type, [1, IN_E], name_hint=f"flex_in{i}") for i in range(2)]
    w_s = [b.smem_alloc(io_type, [1, W_E], name_hint=f"flex_w{i}") for i in range(2)]
    in_a = [b.smem_addr_of(v) for v in in_s]
    w_a = [b.smem_addr_of(v) for v in w_s]

    def dma(cur, st: int):
        """This wave's NXI input + NWI weight DMAs of the cursor's step into stage st."""
        live = b.cmp_lt(cur["t"], TI)
        dh = b.mul(cur["r"], ci(DH))
        dw = b.mul(cur["s"], ci(DW))
        toff = b.mul(
            b.add(b.mul(b.add(b.mul(dh, p_Wi), dw), C), b.mul(cur["ch"], ci(_CK))), c2
        )
        xr = b.buffer_rsrc(params["A"], b.select(live, params["A_bytes"], c0))
        tap = b.add(b.mul(cur["r"], ci(S)), cur["s"])
        sh_t = b.sub(ci(31), tap)
        for i in range(NXI):
            ok = b.cmp_lt(b.shl(cur[f"vm{i}"], sh_t), c0)  # bit tap set
            voff = b.select(ok, b.add(cur[f"xb{i}"], toff), oob)
            dst = b.smem_ptr_add(
                in_a[st],
                b.zext(b.mul(b.add(b.mul(wave, ci(NXI)), ci(i)), ci(1024)), I64),
            )
            b.async_buffer_load_lds(xr, dst, voff, c0, 4)
        woff = b.mul(b.add(b.mul(tap, C), b.mul(cur["ch"], ci(_CK))), c2)
        wr = b.buffer_rsrc(params["B"], b.select(live, params["B_bytes"], c0))
        for i in range(NWI):
            dst = b.smem_ptr_add(
                w_a[st],
                b.zext(b.mul(b.add(b.mul(wave, ci(NWI)), ci(i)), ci(1024)), I64),
            )
            wsh = ((i & 1) * 4 + (i >> 1) * 16) * 2
            b.async_buffer_load_lds(
                wr, dst, cur["wo"], b.add(woff, b.mul(RSC, ci(wsh))), 4
            )

    def frags(st: int):
        ln = opaque_lane()
        r16, q = b.mod(ln, ci(16)), b.div(ln, ci(16))
        a, bb = [], []
        for h in range(2):
            sl = b.mul(b.xor(b.add(ci(h * 4), q), b.mod(r16, ci(8))), ci(8))
            xrow = b.add(b.add(b.mul(g, ci(_BX // 2)), b.mul(wx, ci(64))), r16)
            krow = b.add(b.mul(wk, ci(KT * 16)), r16)
            bb.append(
                [
                    b.smem_load_vN(
                        in_s[st],
                        c0,
                        b.add(b.mul(b.add(xrow, ci(xt * 16)), ci(_CK)), sl),
                        dtype=io_type,
                        n=8,
                    )
                    for xt in range(_XT)
                ]
            )
            a.append(
                [
                    b.smem_load_vN(
                        w_s[st],
                        c0,
                        b.add(b.mul(b.add(krow, ci(kt * 16)), ci(_CK)), sl),
                        dtype=io_type,
                        n=8,
                    )
                    for kt in range(KT)
                ]
            )
        return a, bb

    def mfmas(accs, a, bb):
        accs = list(accs)
        for h in range(2):
            for xt in range(_XT):
                for kt in range(KT):
                    i = xt * KT + kt
                    accs[i] = _mfma(b, dtype, "16x16x32", a[h][kt], bb[h][xt], accs[i])
        return accs

    def bar(prio: int):
        b.s_setprio(prio)
        b.s_barrier_bare()
        b.sched_barrier(0)

    # ---- epilogue -----------------------------------------------------------
    i32x2 = VectorType(I32, 2)
    half8 = VectorType(BF16 if dtype == "bf16" else F16, 8)

    d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])

    def epilogue(accs, t: Value):
        tl = b.div(t, ci(SP))
        tx, tk = b.div(tl, nkt), b.mod(tl, nkt)
        ln = opaque_lane()
        l16, q = b.mod(ln, ci(16)), b.div(ln, ci(16))
        xbase = b.add(
            b.add(b.mul(tx, ci(_BX)), b.mul(g, ci(_BX // 2))),
            b.add(b.mul(wx, ci(64)), l16),
        )
        kbase = b.add(b.mul(tk, ci(BK)), b.mul(wk, ci(KT * 16)))
        if SP == 1:
            kq = b.add(b.mul(b.div(q, c2), ci(16)), b.mul(b.mod(q, c2), ci(8)))
            for xt in range(_XT):
                x = b.add(xbase, ci(xt * 16))
                for cc in range(KT // 2):
                    da = b.vec_bitcast(
                        _trunc_f32(b, dtype, accs[xt * KT + 2 * cc]), i32x2
                    )
                    db = b.vec_bitcast(
                        _trunc_f32(b, dtype, accs[xt * KT + 2 * cc + 1]), i32x2
                    )
                    lo0, hi0 = b.permlane32_swap(
                        b.vec_extract(da, 0), b.vec_extract(db, 0)
                    )
                    lo1, hi1 = b.permlane32_swap(
                        b.vec_extract(da, 1), b.vec_extract(db, 1)
                    )
                    val = b.vec_bitcast(b.vec_pack([lo0, lo1, hi0, hi1], I32), half8)
                    k = b.add(b.add(kbase, ci(cc * 32)), kq)
                    ok = b.land(b.cmp_lt(x, NX), b.cmp_lt(k, K))
                    off = b.select(ok, b.mul(b.add(b.mul(x, K), k), c2), oob)
                    _buf_store_vN(b, dtype, d_rsrc, off, c0, val, 4)
        else:
            pc = b.mod(t, ci(SP))
            slab = b.mul(b.mul(NX, K), ci(4))
            base = b.global_ptr_offset(
                params["D"], b.mul(b.zext(pc, I64), b.zext(slab, I64))
            )
            s_rsrc = b.buffer_rsrc(base, slab)
            kq = b.add(b.mul(b.mod(q, c2), ci(8)), b.mul(b.div(q, c2), ci(4)))
            for xt in range(_XT):
                x = b.add(xbase, ci(xt * 16))
                for kt in range(KT):
                    k = b.add(b.add(kbase, ci(kt * 16)), kq)
                    ok = b.land(b.cmp_lt(x, NX), b.cmp_lt(k, K))
                    off = b.select(ok, b.mul(b.add(b.mul(x, K), k), ci(4)), oob)
                    b.buffer_store_vN_f32(s_rsrc, off, c0, accs[xt * KT + kt], 4)

    # ---- the two groups ------------------------------------------------------
    zero = b.zero_vec_f32(4)
    names = SC + LN

    nacc = _XT * KT
    vec4 = zero.type

    def step(accs, cur, st: int, g1: bool):
        """One step reading stage st (static); returns (accs, cursor)."""
        if not g1:
            dma(cur, 1 - st)
            cur = dict(zip(names, cur_adv(cur)))
            a, bb = frags(st)
            b.s_waitcnt(lgkmcnt=0)
            b.sched_barrier(0)
            bar(1)
            accs = mfmas(accs, a, bb)
            b.sched_barrier(0)
            b.s_waitcnt(vmcnt=0)
            bar(0)
        else:
            a, bb = frags(st)
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sched_barrier(0)
            bar(1)
            dma(cur, st)  # step + 2 into this stage (read by both groups)
            cur = dict(zip(names, cur_adv(cur)))
            accs = mfmas(accs, a, bb)
            b.sched_barrier(0)
            bar(0)
        return accs, cur

    def item_steps(accs, cur, steps: Value, p: int, g1: bool, tag: str):
        """All steps of an item that starts on stage p: pairs, then an odd tail."""
        loop = b.scf_for_iter(
            c0,
            b.div(steps, c2),
            c1,
            [(f"{tag}a{i}", v) for i, v in enumerate(accs)]
            + [(f"{tag}c_{k}", cur[k]) for k in names],
            iv_name=f"{tag}pair",
            elide_trailing_barrier=False,
        )
        with loop as (_, lc):
            la = list(lc[:nacc])
            lcur = dict(zip(names, lc[nacc:]))
            la, lcur = step(la, lcur, p, g1)
            la, lcur = step(la, lcur, 1 - p, g1)
            b.scf_yield(*la, *[lcur[k] for k in names])
        res = list(loop.results)
        accs, cur = res[:nacc], dict(zip(names, res[nacc:]))
        tail = b.scf_if_else(
            b.cmp_eq(b.mod(steps, c2), c1), [vec4] * nacc + [I32] * (len(names) + 1)
        )
        with tail as (then_ctx, else_ctx):
            with then_ctx:
                ta, tcur = step(accs, cur, p, g1)
                b.scf_yield(*ta, *[tcur[k] for k in names], ci(1 - p))
            with else_ctx:
                b.scf_yield(*accs, *[cur[k] for k in names], ci(p))
        res = list(tail.results)
        return res[:nacc], dict(zip(names, res[nacc:-1])), res[-1]

    def run(g1: bool):
        cur = cursor_at(c0)
        dma(cur, 0)
        cur = dict(zip(names, cur_adv(cur)))
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        if g1:
            b.s_barrier_bare()  # group 1 runs one barrier behind
            dma(cur, 1)
            cur = dict(zip(names, cur_adv(cur)))
        b.sched_barrier(0)
        outer = b.scf_for_iter(
            c0,
            rounds,
            c1,
            [(f"o_{k}", cur[k]) for k in names] + [("o_st", c0)],
            iv_name="rnd",
            elide_trailing_barrier=False,
        )
        with outer as (rnd, oc):
            cur = dict(zip(names, oc[: len(names)]))
            st = oc[len(names)]
            t = tile_of(rnd)
            info = item_info(t)
            steps = b.mul(
                b.mul(b.sub(info["r1"], info["r0"]), b.sub(info["s1"], info["s0"])),
                b.sub(info["ch1"], info["ch0"]),
            )
            accs = [zero] * nacc
            par = b.scf_if_else(
                b.cmp_eq(st, c0), [vec4] * nacc + [I32] * (len(names) + 1)
            )
            with par as (then_ctx, else_ctx):
                for p, ctx in ((0, then_ctx), (1, else_ctx)):
                    with ctx:
                        pa, pcur, pst = item_steps(accs, cur, steps, p, g1, f"p{p}")
                        b.scf_yield(*pa, *[pcur[k] for k in names], pst)
            res = list(par.results)
            accs, cur, st = res[:nacc], dict(zip(names, res[nacc:-1])), res[-1]
            epilogue(accs, t)
            b.sched_barrier(0)
            b.scf_yield(*[cur[k] for k in names], st)
        b.s_setprio(0)
        if not g1:
            b.s_barrier_bare()  # balance the barrier count

    with b.scf_if_else(b.cmp_eq(g, c0)) as (then_ctx, else_ctx):
        with then_ctx:
            run(False)
        with else_ctx:
            run(True)
    return b.kernel
