# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Patch-embedding dgrad on wave32 WMMA (gfx1250).

A patch embedding is a convolution whose stride equals its (square) filter
``P`` and has no padding: no two dY pixels touch the same dX pixel. Its dgrad
is therefore one dense GEMM with no reduction across taps::

    dX[n, i*P + r, j*P + s, c] = sum_k dY[n, i, j, k] * W[k, r, s, c]

    M = dY pixels (n, i, j)      N_gemm = P*P*C (r, s, c)      K = dY channels

The KRSC weight tensor already is the row-major ``[K][P*P*C]`` B operand, so
there is no weight pre-pass. Per workgroup::

    tile_h x tile_w             dY pixels         (waves_p * tiles_p * 16)
    r_blk patch rows            r_blk*P*C GEMM columns, padded to whole
                                16-column tiles  (waves_c * tiles_c * 16)
    ck                          dY channels reduced per LDS stage

    for k0 in range(0, K, ck):                    # runtime scf.for
        TDM dY[i0 : i0 + tile_h, j0 : j0 + tile_w, k0 : k0 + ck] -> LDS [i][j][k]
        TDM W[k0 : k0 + ck, r0*P*C : (r0 + r_blk)*P*C]            -> LDS [k][col]
        acc[col-tile, px-tile] = wmma_16x16x32(W^T fragment, dY fragment, acc)

The operand staging is the direct WMMA dgrad's (``conv_direct_dgrad_wmma``)
with a 1x1 filter. The epilogue does the pixel shuffle: the GEMM column
``(r, s, c)`` of dY pixel ``(i, j)`` lands in dX row ``i*P + r``, at the
``s*C + c``-th element of the ``P*C`` run that starts at column ``j*P``. So a
block's dX tile is ``tile_h x r_blk`` dX rows of ``tile_w*P*C`` contiguous
elements each: it is staged in LDS packed as [i][r][j][s, c] and stored with
one rank-3 TDM transfer (dim 0 the row run, dim 1 the patch row, dim 2 the
dY row), clipped to the image.

When ``P`` does not divide the image, the last ``H - Ho*P`` rows and
``W - Wo*P`` columns of dX get no gradient; ``N`` extra workgroups at the end
of the grid zero them.

AOT: batch, extents and the dY channel count are kernargs; the patch size,
the dX channel count ``C`` (it shapes the shuffle) and the tile geometry are
baked.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import (
    CodegenPolicy,
    apply_codegen_policy,
    normalize_scheduler_strategy,
)
from rocke.core.ir import I32, IRBuilder, KernelDef, Value

from kernels.common.conv_direct_dgrad_wmma import _tdm_descriptor_3d
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    emit_direct_params,
)

WMMA_K = 32
CH_PAD = 8
PATCH_WMMA_VGPR_BUDGET = 448
_I32_MAX = (1 << 31) - 1


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def is_patch_dgrad_problem(p: DirectConvProblem) -> bool:
    """Stride == square filter > 1, no padding or dilation, ungrouped."""
    return (
        p.KH == p.KW
        and p.KH > 1
        and p.stride == p.KH
        and p.PAD == 0
        and p.groups == 1
        and not p.is_dilated
    )


@dataclass(frozen=True)
class DgradPatchWmmaSpec:
    """One configuration of :func:`build_dgrad_patch_wmma`."""

    problem: DirectConvProblem
    name: str = "dgrad_patch_wmma"
    waves_c: int = 4  # waves along GEMM columns (r, s, c)
    waves_p: int = 2  # waves along dY pixels
    tiles_c: int = 3  # 16-column tiles per wave
    tiles_p: int = 4  # 16-pixel tiles per wave
    tile_w: int = 16  # dY columns per block; tile_h follows from the pixel count
    r_blk: int = 4  # patch rows per block
    ck: int = 32  # dY channels per LDS stage
    stages: int = 2
    prefetch: bool = False
    sched: str | None = None
    wave_size: int = 32

    @property
    def P(self) -> int:
        return self.problem.KH

    @property
    def run(self) -> int:  # elements of one patch row: P*C
        return self.P * self.problem.cpg

    @property
    def block_cols(self) -> int:  # real GEMM columns of a block
        return self.r_blk * self.run

    @property
    def block_c(self) -> int:
        return self.waves_c * self.tiles_c * 16

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def tile_h(self) -> int:
        return self.block_p // self.tile_w if self.tile_w > 0 else 0

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
    def x_elems(self) -> int:
        return self.block_p * self.x_pitch

    @property
    def wei_elems(self) -> int:
        return self.ck * self.w_pitch

    def lds_bytes(self) -> int:
        ring = self.stages * (self.x_elems + self.wei_elems)
        epi = self.block_p * self.block_cols
        return 2 * max(ring, epi)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"p{self.P}",
            f"c{p.cpg}",
            f"wc{self.waves_c}",
            f"wp{self.waves_p}",
            f"tc{self.tiles_c}",
            f"tp{self.tiles_p}",
            f"tw{self.tile_w}",
            f"rb{self.r_blk}",
            f"ck{self.ck}",
            f"st{self.stages}",
            "pf" if self.prefetch else "",
            f"s{self.sched}" if self.sched else "",
            flags={"bf16": p.dtype == "bf16"},
        )

    def main_blocks(self) -> int:
        p = self.problem
        return (
            p.N
            * _ceil_div(p.Ho, self.tile_h)
            * _ceil_div(p.Wo, self.tile_w)
            * (self.P // self.r_blk)
        )

    def has_border(self) -> bool:
        p = self.problem
        return p.Ho * self.P < p.H or p.Wo * self.P < p.W

    def grid(self) -> tuple[int, int, int]:
        extra = self.problem.N if self.has_border() else 0
        return (self.main_blocks() + extra, 1, 1)


def dgrad_patch_tile_efficiency(spec: DgradPatchWmmaSpec) -> float:
    p = spec.problem
    th, tw = spec.tile_h, spec.tile_w
    pix = (p.Ho * p.Wo) / (_ceil_div(p.Ho, th) * th * _ceil_div(p.Wo, tw) * tw)
    return pix * spec.block_cols / spec.block_c


def dgrad_patch_reason(spec: DgradPatchWmmaSpec, arch: str = "gfx1250") -> str | None:
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return str(e)
    p = spec.problem
    if target.wave_size != 32 or spec.wave_size != 32:
        return "dgrad_patch_wmma is a wave32 kernel"
    if not target.memory.has_tdm:
        return f"no TDM on {arch}"
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return f"missing wmma_f32_16x16x32_{ab} on {arch}"
    if not is_patch_dgrad_problem(p):
        return "not a patch embedding (stride == square filter, pad 0, ungrouped)"
    if p.kpg % 8:
        return f"K {p.kpg} must be a multiple of 8 (16-byte dY rows)"
    if spec.ck % WMMA_K or spec.ck <= 0:
        return f"ck {spec.ck} must be a positive multiple of {WMMA_K}"
    if min(spec.waves_c, spec.waves_p, spec.tiles_c, spec.tiles_p, spec.r_blk) < 1:
        return "waves, tiles and r_blk must be >= 1"
    if spec.P % spec.r_blk:
        return f"r_blk {spec.r_blk} must divide P {spec.P}"
    pad_cols = spec.block_c - spec.block_cols
    if pad_cols < 0 or pad_cols > max(15, spec.block_c // 8):
        return (
            f"block_c {spec.block_c} must cover r_blk*P*C = {spec.block_cols} "
            "with at most an eighth of padding"
        )
    if spec.stages < 2:
        return "stages must be >= 2"
    if spec.tile_w < 1 or spec.block_p % spec.tile_w:
        return f"tile_w {spec.tile_w} must divide {spec.block_p}"
    if spec.waves_c * spec.waves_p < 2:
        return "needs at least two waves"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if spec.tile_w * spec.run > 0xFFFF or spec.w_pitch > 0xFFFF:
        return "tile too wide for a TDM descriptor"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    vgprs = 8 * spec.tiles_c * spec.tiles_p + 8 * (spec.tiles_c + spec.tiles_p) * (
        2 if spec.prefetch else 1
    )
    if vgprs > PATCH_WMMA_VGPR_BUDGET:
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


def is_valid_dgrad_patch_spec(
    spec: DgradPatchWmmaSpec, arch: str = "gfx1250"
) -> tuple[bool, str]:
    why = dgrad_patch_reason(spec, arch)
    return (why is None, why or "ok")


def build_dgrad_patch_wmma(
    spec: DgradPatchWmmaSpec, arch: str = "gfx1250"
) -> KernelDef:
    why = dgrad_patch_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid DgradPatchWmmaSpec for {arch}: {why}")

    p = spec.problem
    P, C = spec.P, p.cpg
    RB = spec.r_blk
    io_type = _io_type(p.dtype)
    ab = "bf16" if p.dtype == "bf16" else "f16"
    wmma_op = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=WMMA_K
    )

    WAVE = spec.wave_size
    WC = spec.waves_c
    THREADS = spec.threads_per_block
    TH, TW = spec.tile_h, spec.tile_w

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    params = emit_direct_params(b, direction="dgrad", io_type=io_type)
    D = params["D"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c16 = b.const_i32(16)
    c_N = params["p_N"]
    c_H = params["p_Hi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]

    tid = b.thread_id_x()
    c_wave = b.const_i32(WAVE)
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_c = b.mod(wave_id, b.const_i32(WC))
    wave_p = b.div(wave_id, b.const_i32(WC))

    n_rt = P // RB
    n_ht = b.div(b.add(c_Ho, b.const_i32(TH - 1)), b.const_i32(TH))
    n_wt = b.div(b.add(c_Wo, b.const_i32(TW - 1)), b.const_i32(TW))
    n_main = b.mul(b.mul(c_N, n_ht), b.mul(n_wt, b.const_i32(n_rt)))
    bid = b.block_id_x()

    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="pt_lds")
    lds_addr = b.trunc(b.smem_addr_of(lds), I32)

    with b.scf_if_else(b.cmp_lt(bid, n_main)) as (main_ctx, zero_ctx):
        with main_ctx:
            _main(
                b,
                spec,
                params,
                bid,
                lds,
                lds_addr,
                wave_id,
                wave_c,
                wave_p,
                lane,
                l16,
                lhalf,
                tid,
                n_ht,
                n_wt,
                wmma_op,
                io_type,
            )
        with zero_ctx:
            # Zero the dX rows and columns no patch covers, image bid - n_main.
            n_img = b.sub(bid, n_main)
            img = b.mul(n_img, params["p_dX_stride_n"])
            row = params["p_dX_stride_hi"]  # W*C
            hc = b.mul(c_Ho, b.const_i32(P))
            wc = b.mul(b.mul(c_Wo, b.const_i32(P)), b.const_i32(C))
            zero = b.vec_extract(b.zero_vec(io_type, 2), 0)
            # Bottom rows: one contiguous run.
            n_bot = b.mul(b.sub(c_H, hc), row)
            bot0 = b.add(img, b.mul(hc, row))
            with b.scf_for(
                c0,
                b.div(b.add(n_bot, b.const_i32(THREADS - 1)), b.const_i32(THREADS)),
                c1,
                iv_name="zb",
            ) as it:
                e = b.add(b.mul(it, b.const_i32(THREADS)), tid)
                with b.scf_if(b.cmp_lt(e, n_bot)):
                    b.global_store(D, b.add(bot0, e), zero, align=2)
            # Right columns of the covered rows.
            rw = b.sub(row, wc)
            n_right = b.mul(hc, rw)
            with b.scf_if(b.cmp_gt(rw, c0)):
                with b.scf_for(
                    c0,
                    b.div(
                        b.add(n_right, b.const_i32(THREADS - 1)), b.const_i32(THREADS)
                    ),
                    c1,
                    iv_name="zr",
                ) as it:
                    e = b.add(b.mul(it, b.const_i32(THREADS)), tid)
                    with b.scf_if(b.cmp_lt(e, n_right)):
                        rr = b.div(e, rw)
                        cc = b.mod(e, rw)
                        off = b.add(img, b.add(b.mul(rr, row), b.add(wc, cc)))
                        b.global_store(D, off, zero, align=2)
    return b.kernel


def _main(
    b,
    spec,
    params,
    bid,
    lds,
    lds_addr,
    wave_id,
    wave_c,
    wave_p,
    lane,
    l16,
    lhalf,
    tid,
    n_ht,
    n_wt,
    wmma_op,
    io_type,
):
    P = spec.P
    RUN = spec.run
    NG = P * RUN
    RB = spec.r_blk
    BCR = spec.block_cols
    TM, TN = spec.tiles_c, spec.tiles_p
    TH, TW = spec.tile_h, spec.tile_w
    CK, NS = spec.ck, spec.stages
    XP, WPITCH = spec.x_pitch, spec.w_pitch
    XE, WEI = spec.x_elems, spec.wei_elems
    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    c_K = params["p_total_k"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]
    A, Bp, D = params["A"], params["B"], params["D"]

    # bid = ((n * n_ht + ht) * n_wt + wt) * n_rt + rt: the patch-row blocks of
    # one dY tile run next to each other and share its dY reads in L2.
    n_rt = P // RB
    t = bid
    rt = b.mod(t, b.const_i32(n_rt))
    t = b.div(t, b.const_i32(n_rt))
    wt = b.mod(t, n_wt)
    t = b.div(t, n_wt)
    ht = b.mod(t, n_ht)
    n_img = b.div(t, n_ht)
    h_tile = b.mul(ht, b.const_i32(TH))
    w_tile = b.mul(wt, b.const_i32(TW))
    r0 = b.mul(rt, b.const_i32(RB))
    c_CK = b.const_i32(CK)
    n_chunks = b.div(b.add(c_K, b.const_i32(CK - 1)), c_CK)

    WEI0 = NS * XE
    wei_addr = b.add(lds_addr, b.const_i32(WEI0 * 2))
    dy_base = b.add(
        b.add(
            b.mul(n_img, params["p_dY_stride_n"]),
            b.mul(h_tile, params["p_dY_stride_ho"]),
        ),
        b.mul(w_tile, params["p_dY_stride_wo"]),
    )
    dy_w_ext = b.smax(b.smin(b.sub(c_Wo, w_tile), b.const_i32(TW)), c0)
    dy_h_ext = b.smax(b.sub(c_Ho, h_tile), c0)
    w_base = b.mul(r0, b.const_i32(RUN))
    c_NG = b.const_i32(NG)

    def _issue(chunk: Value) -> None:
        stage = b.mod(chunk, b.const_i32(NS))
        k0 = b.mul(chunk, c_CK)
        k_ext = b.smax(b.smin(b.sub(c_K, k0), c_CK), c0)
        with b.scf_if(b.cmp_eq(wave_id, c0)):
            src = b.global_ptr_add(A, b.mul(b.add(dy_base, k0), c2))
            dst = b.add(lds_addr, b.mul(stage, b.const_i32(XE * 2)))
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(k_ext, dy_w_ext, dy_h_ext),
                    strides=(params["p_dY_stride_wo"], params["p_dY_stride_ho"]),
                    tile0=XP,
                    tile1=TW,
                    tile2=b.const_i32(TH),
                )
            )
        with b.scf_if(b.cmp_eq(wave_id, c1)):
            src = b.global_ptr_add(Bp, b.mul(b.add(w_base, b.mul(k0, c_NG)), c2))
            dst = b.add(wei_addr, b.mul(stage, b.const_i32(WEI * 2)))
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(b.const_i32(BCR), k_ext, c1),
                    strides=(c_NG, b.mul(c_NG, c_CK)),
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

    # ---- epilogue: stage as [i][r][j][s, c] and store with one rank-3 TDM.
    b.sync_lds_only()
    vec_ok = RUN % 8 == 0
    for m in range(TM):
        col = b.add(
            b.mul(b.add(b.mul(wave_c, b.const_i32(TM)), b.const_i32(m)), c16),
            b.mul(lhalf, c8),
        )
        for tn in range(TN):
            px = b.add(
                b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
            )
            pi = b.div(px, b.const_i32(TW))
            pj = b.mod(px, b.const_i32(TW))
            val = b.vec_cast_f32_to(accs[m * TN + tn], io_type)

            def _addr(n_col):
                rl = b.div(n_col, b.const_i32(RUN))
                q = b.mod(n_col, b.const_i32(RUN))
                return b.add(
                    b.mul(
                        b.add(
                            b.mul(
                                b.add(b.mul(pi, b.const_i32(RB)), rl), b.const_i32(TW)
                            ),
                            pj,
                        ),
                        b.const_i32(RUN),
                    ),
                    q,
                )

            if vec_ok:
                with b.scf_if(b.cmp_lt(col, b.const_i32(BCR))):
                    b.smem_store_vN(lds, [c0, _addr(col)], val, n=8)
            else:
                for e in range(8):
                    n_col = b.add(col, b.const_i32(e))
                    with b.scf_if(b.cmp_lt(n_col, b.const_i32(BCR))):
                        b.smem_store_vN(
                            lds, [c0, _addr(n_col)], b.vec_extract(val, e), n=1
                        )
    b.sync_lds_only()
    with b.scf_if(b.cmp_eq(wave_id, c0)):
        row = params["p_dX_stride_hi"]  # W*C
        off = b.add(
            b.add(
                b.mul(n_img, params["p_dX_stride_n"]),
                b.mul(b.add(b.mul(h_tile, b.const_i32(P)), r0), row),
            ),
            b.mul(b.mul(w_tile, b.const_i32(P)), params["p_dX_stride_wi"]),
        )
        dst = b.global_ptr_add(D, b.mul(off, c2))
        b.tensor_store_from_lds(
            *_tdm_descriptor_3d(
                b,
                dst,
                lds_addr,
                dims=(
                    b.mul(dy_w_ext, b.const_i32(RUN)),
                    b.const_i32(RB),
                    dy_h_ext,
                ),
                strides=(row, b.mul(row, b.const_i32(P))),
                tile0=TW * RUN,
                tile1=RB,
                tile2=b.const_i32(TH),
            )
        )
        b.s_wait_tensorcnt(0)
