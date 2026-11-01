# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.
"""FlyDSL backward prep for the fp32 split backward (gfx950): Delta and the workspace LSE'.

The fp32 twin of `fmha_bwd_prep_gfx950` (same math, same ABI, same addressing; no sink twin: fp32 sinks are out of scope). Per query row:

    Delta[row]  = sum_d O[row, d] * dO[row, d]        fp32 accumulate
    LSE'[row]   = LSE[row] with -inf -> +inf          (NaN and finite values unchanged)

The dK/dV and dQ kernels read `exp2(s - lse * log2e)`: an empty row (LSE = -inf, what hipDNN writes) would give `exp2(+inf)`, so they
need +inf. The input LSE is never written; LSE' goes to its own workspace.

**Layout.** As the 16-bit prep: Delta and LSE' are written through `fmha.lse_row_addressing` / `fmha.lse_token_pitch`, the decode the
dK/dV and dQ kernels read with (dense (H,T), CUMULATIVE and INDIVIDUAL+ARRAY varlen, TH (T,H) LSE). Rows past a sequence's length are
neither read nor written; grid = (cdiv(max_seqlen_q, ROWS), num_head_q, num_seqlens or batch_size). O and dO are `(batch, head, row,
col)` with unit column stride and (batch, head, seq) strides in elements (4-byte elements).

**Reduction.** 128-bit loads of 4 fp32 (`VEC = 4`), fp32 products, in-register `reduce(ADD)`, then `shuffle_xor` across the lanes of a
row group. A row owns `TPR = gcd(tile / 4, 16)` lanes (a power of two <= 16, so a group never crosses a wave); no LDS. Rows per block =
256 / TPR. The lane order is fixed, so Delta is bitwise repeatable. Inputs are already fp32: there is no widening step.

**Exact vs padded.** An exact build requires hdim_vo == tile and uses 4-element vector loads. A padded build reads `hdim_vo <= tile`
columns element by element with the tail masked to zero, for any hdim_vo.

ABI (identical to `fmha_bwd_prep_gfx950`):
    O, DO, LSE (in), LSE_ws (out), Delta (out), batch_size, seqinfo_q0, seqinfo_q1, varlen_bits, num_seqlens,
    max_seqlen_q, num_head_q, hdim_vo, stride_o_{batch,head,seq}, stride_do_{batch,head,seq}
"""

import math

from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "fmha_bwd_prep_f32_gfx950"
BLOCK_THREADS = 256
WARP_SIZE = 64
VEC = 4  # fp32 elements per 128-bit access

# No fast-math hints: -inf must survive the LSE select and the reduction must stay IEEE.
_COMPILE_HINTS = {}


def prep_threads_per_row(tile):
    """Lanes per row: gcd(tile / 4, 16). A power of two <= 16, so a row group stays inside one wave."""
    return math.gcd(tile // VEC, 16)


def prep_arg_values(vals, O, LSE_ws, delta=None):
    """`vals` for the prep launcher: the dict of launch values keyed by kernel argument name (it already holds DO,
    LSE, Delta, varlen descriptors, strides). Adds `O`, `LSE_ws` and `stride_o_*` (O shares dO's layout).
    """
    out = dict(vals)
    out["O"], out["LSE_ws"] = O, LSE_ws
    if delta is not None:
        out["Delta"] = delta
    for a in ("batch", "head", "seq"):
        out[f"stride_o_{a}"] = vals[f"stride_do_{a}"]
    return out


def build_fmha_bwd_prep_f32_gfx950_module_primary(dtype_str, tile, padded):
    if dtype_str != "fp32":
        raise ValueError(dtype_str)
    if tile % 32 or tile <= 0:
        raise ValueError(f"tile {tile}")
    elem = fx.Float32
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
    def fmha_bwd_prep_f32_gfx950_kernel(
        O: fx.Pointer,
        DO: fx.Pointer,
        LSE: fx.Pointer,
        LSE_ws: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
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

        seqlen, row_off, batch = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z
        )
        tokens = fmha.lse_token_pitch(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens
        )
        lse_base, lse_pitch = fmha.lse_row_addressing(
            varlen_bits, batch, head, fx.Index(num_head_q), fx.Index(tokens), row_off
        )

        valid = row < seqlen
        # An out-of-range row reads row 0 of its sequence (always in bounds) and stores nothing.
        row_c = valid.select(row, fx.Int32(0))
        abs_row = fx.Index(row_off) + fx.Index(row_c)
        o_row = (
            fx.Index(batch) * fx.Index(stride_o_batch)
            + head * fx.Index(stride_o_head)
            + abs_row * fx.Index(stride_o_seq)
        )
        do_row = (
            fx.Index(batch) * fx.Index(stride_do_batch)
            + head * fx.Index(stride_do_head)
            + abs_row * fx.Index(stride_do_seq)
        )

        o_p = fx.recast_iter(_ptr_ty(elem, 4), O)
        do_p = fx.recast_iter(_ptr_ty(elem, 4), DO)

        acc = fx.Float32(0.0)
        for i in range_constexpr(NVEC):
            col = (lane + fx.Int32(i * TPR)) * fx.Int32(VEC)
            if const_expr(PADDED):
                # element-wise, tail masked: the column is clamped to 0 (legal) and the value zeroed
                for j in range_constexpr(VEC):
                    c = col + fx.Int32(j)
                    ok = c < fx.Int32(hdim_vo)
                    cc = ok.select(c, fx.Int32(0))
                    ov = fx.ptr_load(o_p + fx.Int64(o_row + fx.Index(cc)), elem)
                    dv = fx.ptr_load(do_p + fx.Int64(do_row + fx.Index(cc)), elem)
                    acc = acc + ok.select(ov * dv, fx.Float32(0.0))
            else:
                vt = Vec.make_type(VEC, elem)
                ov = fx.ptr_load(o_p + fx.Int64(o_row + fx.Index(col)), vt)
                dv = fx.ptr_load(do_p + fx.Int64(do_row + fx.Index(col)), vt)
                acc = acc + (ov * dv).reduce(ReductionOp.ADD)

        for sh in range_constexpr(LOG2_TPR):
            acc = acc + gpu.shuffle_xor(acc, TPR >> (sh + 1), WARP_SIZE)

        if (lane == fx.Int32(0)) & valid:
            off = lse_base + fx.Index(row) * lse_pitch
            f32 = _ptr_ty(fx.Float32, 4)
            lse_v = fx.ptr_load(fx.recast_iter(f32, LSE) + fx.Int64(off), fx.Float32)
            inf = fx.Float32(float("inf"))
            neg_inf = lse_v == fx.Float32(float("-inf"))
            fx.ptr_store(
                neg_inf.select(inf, lse_v), fx.recast_iter(f32, LSE_ws) + fx.Int64(off)
            )
            fx.ptr_store(acc, fx.recast_iter(f32, Delta) + fx.Int64(off))

    @flyc.jit
    def launch_fmha_bwd_prep_f32_gfx950(
        O: fx.Pointer,
        DO: fx.Pointer,
        LSE: fx.Pointer,
        LSE_ws: fx.Pointer,
        Delta: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
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
        fmha_bwd_prep_f32_gfx950_kernel(
            O,
            DO,
            LSE,
            LSE_ws,
            Delta,
            seqinfo_q0,
            seqinfo_q1,
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
        return flyc.compile(launch_fmha_bwd_prep_f32_gfx950, *args, **kwargs)

    _launch.tpr = TPR
    return _launch
