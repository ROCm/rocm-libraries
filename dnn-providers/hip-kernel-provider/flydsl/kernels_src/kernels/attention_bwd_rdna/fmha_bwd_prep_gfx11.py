# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""FlyDSL backward prep for the split backward (gfx11-generic / gfx12-generic): Delta and the workspace LSE'.

The gfx950 prep (`fmha_bwd_prep_gfx950.py`) with `WARP_SIZE = 32` and no sink twin (`sink` is a gfx950 knob). Per query row:

    Delta[row]  = sum_d O[row, d] * dO[row, d]        fp32 accumulate
    LSE'[row]   = LSE[row] with -inf -> +inf          (NaN and finite values unchanged)

The dK/dV and dQ kernels read `exp2(s - lse * log2e)`: an empty row (LSE = -inf) would give `exp2(+inf)`, so they need +inf.
The input LSE is never written; LSE' goes to its own workspace.

**Wave size.** The only wave-size-dependent site is the `shuffle_xor` width (`WARP_SIZE`). A row owns TPR = gcd(tile / 8, 16)
lanes, a power of two <= 16, so the xor-tree offsets TPR/2 .. 1 stay inside one 32-lane wave and the tree is the same
pairing as on wave64. Rows per block = 256 / TPR; the 256-thread block is 8 waves. No LDS.

**Layout, exact vs padded** as in the gfx950 file: an exact build requires hdim_vo == tile (8-element vector loads), a padded build reads
`hdim_vo <= tile` columns element by element with the tail masked to zero. O, dO and the caller's LSE are each addressed from
their own origin (`fmha.tensor_origin_gated`: `off[z] * mul` when ragged, `z * stride_batch` when dense; LSE with its three element strides);
Delta and LSE' go to the compact `[B, H, max_seqlen_q]` workspace (`fmha.stats_ws_base`).

ABI = the gfx950 plain prep launcher's (argument names, order and annotations); the launcher keeps its gfx950 name:
    O, DO, LSE (in), LSE_ws (out), Delta (out), batch_size, seqinfo_q0, seqinfo_q1,
    off_O, mul_O, off_DO, mul_DO, off_Stats, mul_Stats, stride_stats_{batch,head,token},
    varlen_bits, num_seqlens, max_seqlen_q, num_head_q, hdim_vo,
    stride_o_{batch,head,seq}, stride_do_{batch,head,seq}, stream

Derived from `fmha_bwd_prep_gfx950.py` (a port of AOTriton `kernel/bwd_preprocess.py`).
"""

import math


from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "fmha_bwd_prep_gfx11"
BLOCK_THREADS = 256
WARP_SIZE = 32  # RDNA wave32; a row group (TPR <= 16 lanes) never crosses a wave
VEC = 8  # 16-bit elements per 128-bit access

# No fast-math hints: -inf must survive the LSE select and the reduction must stay IEEE.
_COMPILE_HINTS = {}


def prep_threads_per_row(tile):
    """Lanes per row: gcd(tile / 8, 16). A power of two <= 16, so a row group stays inside one wave."""
    return math.gcd(tile // VEC, 16)


def _elem(dtype_str):
    return {"f16": fx.Float16, "bf16": fx.BFloat16}[dtype_str]


def build_fmha_bwd_prep_gfx11_module_primary(dtype_str, tile, padded):
    if dtype_str not in ("f16", "bf16"):
        raise ValueError(dtype_str)
    if tile % 16 or tile <= 0:
        raise ValueError(f"tile {tile}")
    elem = _elem(dtype_str)
    TPR = prep_threads_per_row(tile)
    ROWS = BLOCK_THREADS // TPR
    NVEC = tile // VEC // TPR  # vectors per lane per row
    PADDED = bool(padded)
    LOG2_TPR = TPR.bit_length() - 1
    _cache_tag = (KERNEL_NAME, dtype_str, tile, PADDED)

    def _ptr_ty(ty, align):
        return fx.PointerType.get(
            elem_ty=ty.ir_type, address_space=fx.AddressSpace.Global, alignment=align
        )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def fmha_bwd_prep_gfx11_kernel(
        O: fx.Pointer,
        DO: fx.Pointer,
        LSE: fx.Pointer,
        LSE_ws: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        off_O: fx.Pointer,
        mul_O: fx.Int64,
        off_DO: fx.Pointer,
        mul_DO: fx.Int64,
        off_Stats: fx.Pointer,
        mul_Stats: fx.Int64,
        stride_stats_batch: fx.Int64,
        stride_stats_head: fx.Int64,
        stride_stats_token: fx.Int64,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        num_head_q: fx.Int32,
        hdim_vo: fx.Int32,
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        grp = tid >> fx.Int32(LOG2_TPR)
        lane = tid & fx.Int32(TPR - 1)
        row = fx.Int32(fx.block_idx.x) * fx.Int32(ROWS) + grp
        head = fx.Index(fx.block_idx.y)
        z = fx.Int32(fx.block_idx.z)

        seqlen, _, _ = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z
        )
        o_origin = fmha.tensor_origin_gated(off_O, mul_O, z, stride_o_batch, seqlen)
        do_origin = fmha.tensor_origin_gated(off_DO, mul_DO, z, stride_do_batch, seqlen)
        stats_origin = fmha.tensor_origin_gated(
            off_Stats, mul_Stats, z, stride_stats_batch, seqlen
        )
        ws_base = fmha.stats_ws_base(z, head, num_head_q, max_seqlen_q)

        valid = row < seqlen
        # An out-of-range row reads row 0 of its sequence (always in bounds) and stores nothing.
        row_c = valid.select(row, fx.Int32(0))
        o_row = (
            o_origin
            + head * fx.Index(stride_o_head)
            + fx.Index(row_c) * fx.Index(stride_o_seq)
        )
        do_row = (
            do_origin
            + head * fx.Index(stride_do_head)
            + fx.Index(row_c) * fx.Index(stride_do_seq)
        )

        o_p = fx.recast_iter(_ptr_ty(elem, 2), O)
        do_p = fx.recast_iter(_ptr_ty(elem, 2), DO)

        acc = fx.Float32(0.0)
        if const_expr(PADDED):
            # element-wise, tail masked: the column is clamped to 0 (legal) and the value zeroed. A real loop over the NVEC
            # vector slots, so at most one slot's loads are in flight rather than the whole row's.
            for i_slot, inner in range(
                fx.Index(0), fx.Index(NVEC), 1, init=[fx.as_ir_value(acc)]
            ):
                acc_in = fx.Float32(inner[0])
                col = (lane + fx.Int32(i_slot) * fx.Int32(TPR)) * fx.Int32(VEC)
                for j in range_constexpr(VEC):
                    c = col + fx.Int32(j)
                    ok = c < fx.Int32(hdim_vo)
                    cc = ok.select(c, fx.Int32(0))
                    ov = fx.ptr_load(o_p + fx.Int64(o_row + fx.Index(cc)), elem).to(
                        fx.Float32
                    )
                    dv = fx.ptr_load(do_p + fx.Int64(do_row + fx.Index(cc)), elem).to(
                        fx.Float32
                    )
                    acc_in = acc_in + ok.select(ov * dv, fx.Float32(0.0))
                acc_out = yield [fx.as_ir_value(acc_in)]
            # an scf.for with one carried value hands it back unwrapped
            acc = fx.Float32(acc_out)
        else:
            for i in range_constexpr(NVEC):
                col = (lane + fx.Int32(i * TPR)) * fx.Int32(VEC)
                vt = Vec.make_type(VEC, elem)
                ov = fx.ptr_load(o_p + fx.Int64(o_row + fx.Index(col)), vt).to(
                    fx.Float32
                )
                dv = fx.ptr_load(do_p + fx.Int64(do_row + fx.Index(col)), vt).to(
                    fx.Float32
                )
                acc = acc + (ov * dv).reduce(ReductionOp.ADD)

        for sh in range_constexpr(LOG2_TPR):
            acc = acc + gpu.shuffle_xor(acc, TPR >> (sh + 1), WARP_SIZE)

        if (lane == fx.Int32(0)) & valid:
            stats_off = (
                stats_origin
                + head * fx.Index(stride_stats_head)
                + fx.Index(row) * fx.Index(stride_stats_token)
            )
            ws_off = ws_base + fx.Index(row)
            f32 = _ptr_ty(fx.Float32, 4)
            lse_v = fx.ptr_load(
                fx.recast_iter(f32, LSE) + fx.Int64(stats_off), fx.Float32
            )
            inf = fx.Float32(float("inf"))
            neg_inf = lse_v == fx.Float32(float("-inf"))
            fx.ptr_store(
                neg_inf.select(inf, lse_v),
                fx.recast_iter(f32, LSE_ws) + fx.Int64(ws_off),
            )
            fx.ptr_store(acc, fx.recast_iter(f32, Delta) + fx.Int64(ws_off))

    @flyc.jit
    def launch_fmha_bwd_prep_gfx950(
        O: fx.Pointer,
        DO: fx.Pointer,
        LSE: fx.Pointer,
        LSE_ws: fx.Pointer,
        Delta: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        off_O: fx.Pointer,
        mul_O: fx.Int64,
        off_DO: fx.Pointer,
        mul_DO: fx.Int64,
        off_Stats: fx.Pointer,
        mul_Stats: fx.Int64,
        stride_stats_batch: fx.Int64,
        stride_stats_head: fx.Int64,
        stride_stats_token: fx.Int64,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        num_head_q: fx.Int32,
        hdim_vo: fx.Int32,
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        nblk = (fx.Index(max_seqlen_q) + fx.Index(ROWS - 1)) // fx.Index(ROWS)
        nseq = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        fmha_bwd_prep_gfx11_kernel(
            O,
            DO,
            LSE,
            LSE_ws,
            Delta,
            seqinfo_q0,
            seqinfo_q1,
            off_O,
            mul_O,
            off_DO,
            mul_DO,
            off_Stats,
            mul_Stats,
            stride_stats_batch,
            stride_stats_head,
            stride_stats_token,
            varlen_bits,
            num_seqlens,
            max_seqlen_q,
            num_head_q,
            hdim_vo,
            stride_o_batch,
            stride_o_head,
            stride_o_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
        ).launch(
            grid=(nblk, fx.Index(num_head_q), nseq),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(*args, **kwargs):
        return flyc.compile(launch_fmha_bwd_prep_gfx950, *args, **kwargs)

    _launch.tpr = TPR
    return _launch
