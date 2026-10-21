# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small-C dgrad on wave32 WMMA (gfx1250): GEMM over the flattened filter,
then col2im in LDS.

For a stem convolution (C = 3 or 4 input channels) the dgrad's natural GEMM
(dX channels x pixels) leaves almost all of every MMA tile empty. This kernel
transposes the problem the way the small-C wgrad kernel does: it flattens the
filter taps and the dX channels into one GEMM dimension::

    Z[q, (r, s, c)] = sum_k dY[q, k] * W[k, r, s, c]     q = dY pixel
    dX[h, w, c]     = sum_{(r, s) : h = i*S - PAD + r, w = j*S - PAD + s} Z[(i, j), (r, s, c)]

The KRSC weights are the row-major ``[K][R*S*C]`` B operand already. A
workgroup owns a ``out_h x out_w`` dX tile (both multiples of S) and computes
Z for every dY pixel that reaches it, a ``dy_h x dy_w`` window (the tile's dY
footprint plus its halo; neighbouring blocks recompute the halo pixels, which
costs little on these memory-bound layers). The operand staging is that of the
direct WMMA dgrad (``conv_direct_dgrad_wmma``): per dY channel chunk one TDM
transfer of the dY window (zero rows / columns above / left of the image land
shifted and stay zero) and one of the weights, transposed on the LDS read.

The epilogue stages Z (fp32) in LDS, then each thread forms dX outputs phase
by phase (output row / column mod S): within a phase the contributing taps
are a compile-time list, so the overlap-add is a few LDS reads per output. The
dX tile is staged in LDS and stored with one TDM transfer.

AOT: batch, extents and the dY channel count are kernargs; filter, stride,
padding, the dX channel count and the tile geometry are baked.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import (
    CodegenPolicy,
    apply_codegen_policy,
    normalize_scheduler_strategy,
)
from rocke.core.ir import F32, I32, IRBuilder, KernelDef, Value

from kernels.common.conv_direct_dgrad_wmma import _tdm_descriptor_3d
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    emit_direct_params,
)

WMMA_K = 32
CH_PAD = 8
SMALLC_WMMA_VGPR_BUDGET = 448
SMALLC_MAX_C = 4
_I32_MAX = (1 << 31) - 1


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class DgradSmallcWmmaSpec:
    """One configuration of :func:`build_dgrad_smallc_wmma`."""

    problem: DirectConvProblem
    name: str = "dgrad_smallc_wmma"
    waves_c: int = 1  # waves along GEMM columns (r, s, c)
    waves_p: int = 4  # waves along dY pixels
    tiles_c: int = 2
    tiles_p: int = 5
    out_h: int = 32  # dX rows per block (multiple of the stride)
    out_w: int = 32  # dX columns per block (multiple of the stride)
    ck: int = 32
    stages: int = 2
    prefetch: bool = False
    sched: str | None = None
    wave_size: int = 32

    @property
    def S(self) -> int:
        return self.problem.stride

    @property
    def ncols(self) -> int:  # real GEMM columns: R*S*C
        p = self.problem
        return p.KH * p.KW * p.cpg

    @property
    def off_h(self) -> int:  # dY rows of halo above the tile's first row / S
        p = self.problem
        return (p.KH - 1 - p.PAD) // self.S

    @property
    def off_w(self) -> int:
        p = self.problem
        return (p.KW - 1 - p.PAD) // self.S

    @property
    def dy_h(self) -> int:
        p = self.problem
        return (self.out_h - 1 + p.PAD) // self.S + self.off_h + 1

    @property
    def dy_w(self) -> int:
        p = self.problem
        return (self.out_w - 1 + p.PAD) // self.S + self.off_w + 1

    @property
    def block_c(self) -> int:
        return self.waves_c * self.tiles_c * 16

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def threads_per_block(self) -> int:
        return self.waves_c * self.waves_p * self.wave_size

    @property
    def x_pitch(self) -> int:
        return self.ck + CH_PAD

    @property
    def w_pitch(self) -> int:
        return self.block_c + CH_PAD

    @property
    def z_pitch(self) -> int:  # fp32 Z row (odd-ish skew against bank conflicts)
        return self.block_c + 8

    @property
    def x_elems(self) -> int:
        # The whole GEMM pixel range is read; the window plus the shift slack.
        return (self.block_p + self.off_w + 1) * self.x_pitch

    @property
    def wei_elems(self) -> int:
        return self.ck * self.w_pitch

    def lds_bytes(self) -> int:
        ring = 2 * self.stages * (self.x_elems + self.wei_elems)
        z = 4 * self.block_p * self.z_pitch
        out = 2 * self.out_h * self.out_w * self.problem.cpg
        return max(ring, z + out)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"s{p.stride}",
            f"p{p.PAD}",
            f"c{p.cpg}",
            f"wc{self.waves_c}",
            f"wp{self.waves_p}",
            f"tc{self.tiles_c}",
            f"tp{self.tiles_p}",
            f"oh{self.out_h}",
            f"ow{self.out_w}",
            f"ck{self.ck}",
            f"st{self.stages}",
            "pf" if self.prefetch else "",
            f"s{self.sched}" if self.sched else "",
            flags={"bf16": p.dtype == "bf16"},
        )

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        return (
            p.N * _ceil_div(p.H, self.out_h) * _ceil_div(p.W, self.out_w),
            1,
            1,
        )


def dgrad_smallc_tile_efficiency(spec: DgradSmallcWmmaSpec) -> float:
    p = spec.problem
    oh, ow = spec.out_h, spec.out_w
    return (p.H * p.W) / (_ceil_div(p.H, oh) * oh * _ceil_div(p.W, ow) * ow)


def dgrad_smallc_reason(spec: DgradSmallcWmmaSpec, arch: str = "gfx1250") -> str | None:
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return str(e)
    p = spec.problem
    if target.wave_size != 32 or spec.wave_size != 32:
        return "dgrad_smallc_wmma is a wave32 kernel"
    if not target.memory.has_tdm:
        return f"no TDM on {arch}"
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return f"missing wmma_f32_16x16x32_{ab} on {arch}"
    if p.groups != 1 or p.is_dilated:
        return "ungrouped, undilated convolutions only"
    if p.cpg > SMALLC_MAX_C:
        return f"C {p.cpg} > {SMALLC_MAX_C}"
    if p.stride == p.KH and p.PAD == 0:
        return "patch embedding: use dgrad_patch_wmma"
    if not 0 <= p.PAD <= min(p.KH, p.KW) - 1:
        return f"PAD {p.PAD} outside 0..K-1"
    if p.kpg % 8:
        return f"K {p.kpg} must be a multiple of 8 (16-byte dY rows)"
    if spec.ck % WMMA_K or spec.ck <= 0:
        return f"ck {spec.ck} must be a positive multiple of {WMMA_K}"
    if min(spec.waves_c, spec.waves_p, spec.tiles_c, spec.tiles_p) < 1:
        return "waves and tiles must be >= 1"
    if spec.out_h % spec.S or spec.out_w % spec.S or spec.out_h < 1 or spec.out_w < 1:
        return f"out tile {spec.out_h}x{spec.out_w} must be multiples of S {spec.S}"
    if spec.block_c < spec.ncols or spec.block_c - spec.ncols >= 16:
        return f"block_c {spec.block_c} must be R*S*C = {spec.ncols} rounded up to 16"
    if spec.dy_h * spec.dy_w > spec.block_p:
        return f"dY window {spec.dy_h}x{spec.dy_w} > {spec.block_p} GEMM pixels"
    if spec.stages < 2:
        return "stages must be >= 2"
    if spec.waves_c * spec.waves_p < 2:
        return "needs at least two waves"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    vgprs = 8 * spec.tiles_c * spec.tiles_p + 8 * (spec.tiles_c + spec.tiles_p) * (
        2 if spec.prefetch else 1
    )
    if vgprs > SMALLC_WMMA_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of accumulators and fragments"
    try:
        normalize_scheduler_strategy(spec.sched)
    except ValueError as e:
        return str(e)
    tensor_bytes = (
        2 * p.N * p.Ho * p.Wo * p.total_k,
        2 * p.total_k * p.KH * p.KW * p.cpg,
        2 * p.N * p.H * p.W * p.total_c,
    )
    if max(tensor_bytes) > _I32_MAX:
        return f"tensors of {max(tensor_bytes)} B exceed the i32 byte offsets"
    return None


def is_valid_dgrad_smallc_spec(
    spec: DgradSmallcWmmaSpec, arch: str = "gfx1250"
) -> tuple[bool, str]:
    why = dgrad_smallc_reason(spec, arch)
    return (why is None, why or "ok")


def build_dgrad_smallc_wmma(
    spec: DgradSmallcWmmaSpec, arch: str = "gfx1250"
) -> KernelDef:
    why = dgrad_smallc_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid DgradSmallcWmmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW, S, PAD, C = p.KH, p.KW, p.stride, p.PAD, p.cpg
    NCOL = spec.ncols
    io_type = _io_type(p.dtype)
    ab = "bf16" if p.dtype == "bf16" else "f16"
    wmma_op = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=WMMA_K
    )

    WAVE = spec.wave_size
    WC = spec.waves_c
    TM, TN = spec.tiles_c, spec.tiles_p
    THREADS = spec.threads_per_block
    BP = spec.block_p
    OH, OW = spec.out_h, spec.out_w
    DH, DW = spec.dy_h, spec.dy_w
    OFFH, OFFW = spec.off_h, spec.off_w
    CK, NS = spec.ck, spec.stages
    XP, WPITCH, ZP = spec.x_pitch, spec.w_pitch, spec.z_pitch
    XE, WEI = spec.x_elems, spec.wei_elems

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    params = emit_direct_params(b, direction="dgrad", io_type=io_type)
    A, Bp, D = params["A"], params["B"], params["D"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    c_K = params["p_total_k"]
    c_H, c_W = params["p_Hi"], params["p_Wi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]

    tid = b.thread_id_x()
    c_wave = b.const_i32(WAVE)
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_c = b.mod(wave_id, b.const_i32(WC))
    wave_p = b.div(wave_id, b.const_i32(WC))

    n_ht = b.div(b.add(c_H, b.const_i32(OH - 1)), b.const_i32(OH))
    n_wt = b.div(b.add(c_W, b.const_i32(OW - 1)), b.const_i32(OW))
    t = b.block_id_x()
    wt = b.mod(t, n_wt)
    t = b.div(t, n_wt)
    ht = b.mod(t, n_ht)
    n_img = b.div(t, n_ht)
    h0 = b.mul(ht, b.const_i32(OH))
    w0 = b.mul(wt, b.const_i32(OW))
    c_CK = b.const_i32(CK)
    n_chunks = b.div(b.add(c_K, b.const_i32(CK - 1)), c_CK)

    # The dY window starts at (h0/S - OFFH, w0/S - OFFW); a negative start is
    # clipped and the transfer lands shifted (the cells before it are zeroed).
    i_lo = b.sub(b.div(h0, b.const_i32(S)), b.const_i32(OFFH))
    j_lo = b.sub(b.div(w0, b.const_i32(S)), b.const_i32(OFFW))
    hs = b.smax(b.sub(c0, i_lo), c0)
    sw = b.smax(b.sub(c0, j_lo), c0)
    i_start = b.add(i_lo, hs)
    j_start = b.add(j_lo, sw)
    z_elems = b.mul(b.add(b.mul(hs, b.const_i32(DW)), sw), b.const_i32(XP))
    dy_w_ext = b.smax(b.smin(b.sub(c_Wo, j_start), b.sub(b.const_i32(DW), sw)), c0)
    dy_h_ext = b.smax(b.sub(c_Ho, i_start), c0)
    dy_tile2 = b.sub(b.const_i32(DH), hs)
    dy_base = b.add(
        b.add(
            b.mul(n_img, params["p_dY_stride_n"]),
            b.mul(i_start, params["p_dY_stride_ho"]),
        ),
        b.mul(j_start, params["p_dY_stride_wo"]),
    )
    c_NC = b.const_i32(NCOL)

    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="sc_lds")
    lds_addr = b.trunc(b.smem_addr_of(lds), I32)
    WEI0 = NS * XE
    wei_addr = b.add(lds_addr, b.const_i32(WEI0 * 2))

    max_z = (OFFH * DW + OFFW) * XP
    if max_z:
        zero_v = b.zero_vec(io_type, 8)
        z_vecs = b.div(z_elems, c8)
        for j in range(0, _ceil_div(max_z, 8), THREADS):
            v = b.add(tid, b.const_i32(j))
            with b.scf_if(b.cmp_lt(v, z_vecs)):
                flat = b.mul(v, c8)
                for st in range(NS):
                    b.smem_store_vN(
                        lds, [c0, b.add(flat, b.const_i32(st * XE))], zero_v, n=8
                    )

    def _issue(chunk: Value) -> None:
        stage = b.mod(chunk, b.const_i32(NS))
        k0 = b.mul(chunk, c_CK)
        k_ext = b.smax(b.smin(b.sub(c_K, k0), c_CK), c0)
        with b.scf_if(b.cmp_eq(wave_id, c0)):
            src = b.global_ptr_add(A, b.mul(b.add(dy_base, k0), c2))
            dst = b.add(
                lds_addr,
                b.mul(b.add(b.mul(stage, b.const_i32(XE)), z_elems), c2),
            )
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(k_ext, dy_w_ext, dy_h_ext),
                    strides=(params["p_dY_stride_wo"], params["p_dY_stride_ho"]),
                    tile0=XP,
                    tile1=DW,
                    tile2=dy_tile2,
                )
            )
        with b.scf_if(b.cmp_eq(wave_id, c1)):
            src = b.global_ptr_add(Bp, b.mul(b.mul(k0, c_NC), c2))
            dst = b.add(wei_addr, b.mul(stage, b.const_i32(WEI * 2)))
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(c_NC, k_ext, c1),
                    strides=(c_NC, b.mul(c_NC, c_CK)),
                    tile0=WPITCH,
                    tile1=CK,
                    tile2=c1,
                )
            )

    px_off = []
    for tn in range(TN):
        flat = b.add(
            b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
        )
        px_off.append(b.add(b.mul(flat, b.const_i32(XP)), b.mul(lhalf, c16)))
    tr_col = b.mul(b.div(l16, c8), c8)
    tr_row = b.add(b.mul(lhalf, c16), b.mod(lane, c8))
    wei_off = []
    for m in range(TM):
        col = b.mul(b.add(b.mul(wave_c, b.const_i32(TM)), b.const_i32(m)), c16)
        wei_off.append(
            b.add(
                b.mul(tr_row, b.const_i32(WPITCH)),
                b.add(col, b.add(tr_col, b.const_i32(WEI0))),
            )
        )

    def _read_b(base: Value, tn: int, off: int) -> Value:
        addr = b.add(base, b.add(px_off[tn], b.const_i32(off)))
        lo = b.smem_load_vN(lds, c0, addr, dtype=io_type, n=8)
        hi = b.smem_load_vN(lds, c0, b.add(addr, c8), dtype=io_type, n=8)
        return b.vec_concat(lo, hi)

    def _read_a(base: Value, m: int, off: int) -> Value:
        addr = b.add(base, b.add(wei_off[m], b.const_i32(off)))
        lo = b.ds_read_tr16_b128(lds, c0, addr, dtype=io_type)
        hi = b.ds_read_tr16_b128(
            lds, c0, b.add(addr, b.const_i32(8 * WPITCH)), dtype=io_type
        )
        return b.vec_concat(lo, hi)

    steps = list(range(CK // WMMA_K))

    def _chunk(kc: Value, accs):
        if NS == 2:
            b.s_wait_tensorcnt(0)
        else:
            more = b.cmp_lt(b.add(kc, b.const_i32(NS - 2)), n_chunks)
            with b.scf_if_else(more) as (then_ctx, else_ctx):
                with then_ctx:
                    b.s_wait_tensorcnt(NS - 2)
                with else_ctx:
                    b.s_wait_tensorcnt(0)
        b.sync_lds_only()
        nxt = b.add(kc, b.const_i32(NS - 1))
        with b.scf_if(b.cmp_lt(nxt, n_chunks)):
            _issue(nxt)
        stage = b.mod(kc, b.const_i32(NS))
        xb = b.mul(stage, b.const_i32(XE))
        wb = b.mul(stage, b.const_i32(WEI))

        def _frags(ks):
            bf = [_read_b(xb, tn, ks * WMMA_K) for tn in range(TN)]
            af = [_read_a(wb, m, ks * WMMA_K * WPITCH) for m in range(TM)]
            return af, bf

        accs = list(accs)
        b.s_setprio(1)
        nf = _frags(steps[0]) if spec.prefetch else None
        for j, ks in enumerate(steps):
            if spec.prefetch:
                af, bf = nf
                if j + 1 < len(steps):
                    nf = _frags(steps[j + 1])
            else:
                af, bf = _frags(ks)
            for m in range(TM):
                for tn in range(TN):
                    i = m * TN + tn
                    accs[i] = b.mma(wmma_op, af[m], bf[tn], accs[i])
        b.s_setprio(0)
        return accs

    for st in range(NS - 1):
        with b.scf_if(b.cmp_lt(b.const_i32(st), n_chunks)):
            _issue(b.const_i32(st))
    zero_acc = b.zero_vec_f32(8)
    loop = b.scf_for_iter(
        c0, n_chunks, c1, [(f"acc{i}", zero_acc) for i in range(TM * TN)], iv_name="kc"
    )
    with loop as (kc, carried):
        b.scf_yield(*_chunk(kc, list(carried)))
    accs = list(loop.results)

    # ---- epilogue 1: Z (fp32) to LDS as [pixel][col].
    b.sync_lds_only()
    for m in range(TM):
        col = b.add(
            b.mul(b.add(b.mul(wave_c, b.const_i32(TM)), b.const_i32(m)), c16),
            b.mul(lhalf, c8),
        )
        for tn in range(TN):
            px = b.add(
                b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
            )
            # f32 element i of the shared bf16 allocation sits at index 2*i.
            zi = b.add(b.mul(px, b.const_i32(ZP)), col)
            b.smem_store_vN(lds, [c0, b.mul(zi, c2)], accs[m * TN + tn], n=8)
    b.sync_lds_only()

    # ---- epilogue 2: overlap-add, phase by phase, into a bf16 dX tile.
    OUT0 = BP * ZP * 2  # dX tile after Z, in 2-byte elements
    cvt = b.trunc_f32_to_bf16 if p.dtype == "bf16" else b.trunc_f32_to_f16
    for ph in range(S):
        for pw in range(S):
            taps = [
                (r, s)
                for r in range(KH)
                for s in range(KW)
                if (ph + PAD - r) % S == 0 and (pw + PAD - s) % S == 0
            ]
            na, nb = OH // S, OW // S
            E = na * nb  # output pixels of this phase; a thread does all C
            for it in range(_ceil_div(E, THREADS)):
                e = b.add(tid, b.const_i32(it * THREADS))
                with b.scf_if(b.cmp_lt(e, b.const_i32(E))):
                    bcol = b.mod(e, b.const_i32(nb))
                    arow = b.div(e, b.const_i32(nb))
                    base = b.mul(
                        b.add(b.mul(arow, b.const_i32(DW)), bcol), b.const_i32(ZP)
                    )
                    acc = [b.const_f32(0.0)] * C
                    for r, s in taps:
                        dh = (ph + PAD - r) // S + OFFH
                        dw = (pw + PAD - s) // S + OFFW
                        off = (dh * DW + dw) * ZP + (r * KW + s) * C
                        idx = b.add(base, b.const_i32(off))
                        z = b.smem_load_vN(lds, c0, b.mul(idx, c2), dtype=F32, n=C)
                        acc = [b.fadd(acc[i], b.vec_extract(z, i)) for i in range(C)]
                    oh = b.add(b.mul(arow, b.const_i32(S)), b.const_i32(ph))
                    ow = b.add(b.mul(bcol, b.const_i32(S)), b.const_i32(pw))
                    oidx = b.add(
                        b.mul(b.add(b.mul(oh, b.const_i32(OW)), ow), b.const_i32(C)),
                        b.const_i32(OUT0),
                    )
                    for i in range(C):
                        b.smem_store_vN(
                            lds, [c0, b.add(oidx, b.const_i32(i))], cvt(acc[i]), n=1
                        )
    b.sync_lds_only()
    with b.scf_if(b.cmp_eq(wave_id, c0)):
        off = b.add(
            b.add(
                b.mul(n_img, params["p_dX_stride_n"]),
                b.mul(h0, params["p_dX_stride_hi"]),
            ),
            b.mul(w0, params["p_dX_stride_wi"]),
        )
        dst = b.global_ptr_add(D, b.mul(off, c2))
        b.tensor_store_from_lds(
            *_tdm_descriptor_3d(
                b,
                dst,
                b.add(lds_addr, b.const_i32(OUT0 * 2)),
                dims=(
                    b.mul(b.smax(b.sub(c_W, w0), c0), b.const_i32(C)),
                    b.smax(b.sub(c_H, h0), c0),
                    c1,
                ),
                strides=(
                    params["p_dX_stride_hi"],
                    b.mul(params["p_dX_stride_hi"], c_H),
                ),
                tile0=OW * C,
                tile1=OH,
                tile2=c1,
            )
        )
        b.s_wait_tensorcnt(0)
    return b.kernel
