# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""The dQ backward body at **16 rows per wave**, on `v_mfma_f32_16x16x32_bf16`.

B3.5. A second MFMA family beside the 32-row one in `fmha_bwd_dq_gfx950.py`,
not a replacement: the 32-row family wins below head_dim 256 and this one
exists for the top of the ladder, where B3 left dQ at 303 TF (384) and 110
(512) against ~730 in the middle.

--- Why 16 rows, and why this shape ------------------------------------------

`register_demand` in `fmha_tuning_bwd_dq_gfx950` states the problem exactly. At
32 rows the loop-invariant Q + dO + dQ is `1.0 * head_dim` registers per lane,
which is the whole 512-register file at head_dim 512 *before a single operand*.
At 16 rows it is `0.5 * head_dim`. Nothing else moves that term -- it is
loop-carried, so no amount of staging or scheduling touches it.

The shape follows from the gfx950 scheduling model rather than from the lane
map (`SISchedule.td`, `SIDPGFX950FullSpeedModel`): `16x16x16` is 8192 flops in
4 passes and `32x32x16` is 32768 in 8, so **`16x16x16` is half rate**.
`16x16x32` is 16384 in 4 -- 16 rows at full rate. It is also the cheaper port,
because a lane still holds 8 bf16 of an operand, so `_bf16_trunc_pack_v8` and
the v8 shape of every LDS read carry over.

`BLOCK_N` is **32**, and that is two decisions at once. It closes the last 40
registers at head_dim 512 (552 -> 404 predicted), and it keeps
`BLOCK_N / MFMA_M == 2`, so the score tile is still the `(s_lo, s_hi)` *pair*
that `flash_attn_utils` is written around -- which matters because that file is
shared production code and must not be edited.

--- What is reused, and what could not be ------------------------------------

**The LDS staging is untouched.** `BwdDqKvGmemToLdsLoader` stages K into the V
region and V into the K region exactly as it does for the 32-row family; only
the register reads change. So is the context, the descriptors, the padded-head
contract and the dB store's column map.

What could not be reused is everything shaped by the accumulator, because
`flash_attn_utils` writes `range_constexpr(16)` for "elements in a score half"
and here it is 4: `_score_pair_to_lists`, `_sub_score_pair`,
`_exp2_score_slice`, `_pack_p_v8_slices`. Each is a handful of lines and is
re-spelled below against `ACC_ELEMS`.

--- The one permutation, and where it was put --------------------------------

The score accumulator gives lane `j` (group `g = j // 16`) the KV tokens
`{4g+e}` from `s_lo` and `{16+4g+e}` from `s_hi`. GEMM3's B operand wants 8
contraction values per lane, and the *published* transpose map
(`fmha_mfma16_gfx950`) delivers `k = 8g + i` -- a different set. One of the two
has to bend.

**The transpose read bends**, exactly as it does in the 32-row family, and the
alternative was considered and rejected: permuting GEMM1/GEMM2's token order
instead would leave the accumulator holding a scrambled token map, which the KV
tail mask and the dB store both read. Bending one read keeps the natural map
everywhere else.

The required order is `T(g, 4r + i) = 4g + 16r + i`, which is separable, so it
costs nothing: the group term becomes `4 * (lane // 16)` tokens instead of
`8 * (lane // 16)`, and the second read sits at `+16` tokens instead of `+4`.
`_kt_read_base` below. The result was checked against a host reference before
anything else was written.
"""

from kernels.attention_bwd_cdna import (
    fmha_common_gfx1201 as fmha,
)  # noqa: F401  (kept for the shared row addressing)
from kernels.attention_bwd_cdna.fmha_bwd_bias_exp_gfx950 import FM_BIAS_EXP
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import _fmul, _fsub
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (  # ParityKernelContext documents the base this rides on
    ParityKernelContext,  # noqa: F401
    _ds_read_tr_v4f16_imm,
    _slab_span_elems,
    exp2_wait_state,
    mfma_operand_wait_state,
)
from kernels.attention_bwd_cdna.fmha_mfma16_gfx950 import MFMA16_M
from kernels.attention_bwd_cdna.gfx950_standalone import buffer_ops, dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

__all__ = [
    "M16DqGemmHelper",
    "M16DqReader",
    "M16DqSoftmax",
    "M16DqStore",
    "M16QLoader",
    "acc_elems",
    "d_chunks16",
    "k_steps16",
    "make_m16_dq_body",
    "score_msteps",
]

_s_barrier = dualwave._s_barrier
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier

# The MFMA's contraction extent. 32 here; the 8 bf16 a lane holds of an operand
# are `MFMA_K * MFMA_M / WARP_SIZE`, which is the same 8 the 32-row family uses
# and the reason its packing helpers carry over.
MFMA16_K = 32
LANE_ELEMS = 8


def acc_elems(traits):
    """f32 the accumulator gives one lane per MFMA. 4 here, 16 at 32 rows."""
    return MFMA16_M * MFMA16_M // traits.WARP_SIZE


def score_msteps(traits):
    """MFMA steps along the KV axis of one score tile. **Must be 2.**

    The `(s_lo, s_hi)` pair is not a parameter of `flash_attn_utils`, it is its
    shape -- so `BLOCK_N` is chosen to keep this at 2 rather than the softmax
    path being generalised. Asserted rather than commented, as the rows-per-wave
    ceiling is.
    """
    steps = traits.BLOCK_N // MFMA16_M
    if steps != 2:
        raise ValueError(
            f"BLOCK_N {traits.BLOCK_N} over a {MFMA16_M}-row MFMA gives {steps} score steps, and the "
            "softmax path is written as an (s_lo, s_hi) pair. Pass block_n=32."
        )
    return steps


def k_steps16(traits):
    """MFMA steps along the head dim, for the two d-contracted GEMMs."""
    return traits.HEAD_DIM // MFMA16_K


def d_chunks16(traits):
    """16-wide output chunks of dQ one wave accumulates."""
    return traits.HEAD_DIM // MFMA16_M


class M16DqReader:
    """The three LDS reads this body needs, all from the two staged tiles.

    Instantiated per tile, because the tiles differ in *both* the LDS line
    pitch and the padded-head extent -- the same two-instance split the 32-row
    family uses, and for the same reason.
    """

    def __init__(self, ctx, *, region, line_stride, hdim, hdim_floor):
        self.ctx = ctx
        self.traits = ctx.traits
        self.region = region  # "k" or "v": which of the two staged regions
        self.line = line_stride
        self.hdim = hdim
        self.hdim_floor = int(hdim_floor)
        self.base_elems = (
            dualwave._v_buf_base(self.traits, 0)
            if region == "v"
            else dualwave._k_buf_base(self.traits, 0)
        )
        self.scope = dualwave._dualwave_lds_scope(region, 0)

    # -- addressing ---------------------------------------------------------

    def _tok_off(self, t):
        """`fmha_mfma16_gfx950.tok_off`, over *this* region's pitch.

        The published helper is written against `SMEM_V_LINE_STRIDE` because
        the forward only ever transposes the V region. This kernel reads a
        K-pitch region the same way, so the pitch is a parameter here.
        """
        n = self.traits.SMEM_N_RPT
        return (t // n) * self.traits.D_128B_SIZE + (t % n) * self.line

    def _tok_off_dyn(self, t):
        n = fx.Index(self.traits.SMEM_N_RPT)
        return (t // n) * fx.Index(self.traits.D_128B_SIZE) + (t % n) * fx.Index(
            self.line
        )

    def _d_off(self, d):
        g = self.traits.D_128B_SIZE
        return (d // g) * self.traits.SMEM_N_RPT * self.line + (d % g)

    # -- GEMM1 / GEMM2: A operand with the KV token on M --------------------

    def a_token_base(self, lane):
        """Per-lane base for an A operand whose `m` axis is the KV token.

        `A[m][k]` with `m` the token and `k` the head dim, so lane `l` holds
        token `l % 16` and the eight d columns `8 * (l // 16) + i`. The token
        term is `tok_off(l % 16)` written out: with `SMEM_N_RPT == 4` at
        `BLOCK_N` 32 that is `((l%16)//4) * granule + (l%4) * line`, and the
        M-step then adds a whole `16` tokens, which is `4 * granule` and
        constant.
        """
        n = fx.Index(self.traits.SMEM_N_RPT)
        lane16 = lane % fx.Index(MFMA16_M)
        return (lane16 // n) * fx.Index(self.traits.D_128B_SIZE) + (
            lane16 % n
        ) * fx.Index(self.line)

    def load_a_tokens(self, lane, step, ks):
        """One A-operand pack for (M-step, k-step): 8 contiguous d out of LDS."""
        d0 = LANE_ELEMS * (lane // fx.Index(MFMA16_M)) + fx.Index(ks * MFMA16_K)
        idx = (
            fx.Index(self.base_elems)
            + self.a_token_base(lane)
            + fx.Index(self._tok_off(step * MFMA16_M))
            + self._d_off_dyn(d0)
        )
        return self._lds_pack(idx)

    def _d_off_dyn(self, d):
        g = fx.Index(self.traits.D_128B_SIZE)
        return (d // g) * fx.Index(self.traits.SMEM_N_RPT * self.line) + (d % g)

    def _lds_pack(self, elem_idx):
        traits = self.traits
        ptr = buffer_ops.get_element_ptr(
            self.ctx.lds_kv_base_ptr,
            byte_offset=elem_idx * traits.BF16_BYTES,
            elem_type=T.i8,
        )
        return llvm.LoadOp(
            self.ctx.kv_mfma_pack_type,
            ptr,
            alignment=16,
            alias_scopes=dualwave._dualwave_lds_alias_scopes(self.scope),
            noalias_scopes=dualwave._dualwave_lds_noalias_scopes(
                self.scope, traits.LDS_SCOPE_NAMES
            ),
        ).result

    def load_a_all(self, lane):
        """Every A pack for one tile: `[step][ks]`, padded-head masked.

        Materialised rather than streamed. `register_demand` says one live A
        tile is `BLOCK_N * d / 128` = 128 registers at head_dim 512 with
        `BLOCK_N` 32, which the 16-row budget has room for -- that is the whole
        point of the smaller tile.

        The mask is the 32-row loader's argument transposed onto this lane map:
        masking the *other* operand (Q, dO) is enough for a finite pad because
        `0 * x == 0`, and not enough for a pad holding NaN, which the caller's
        D-axis slack is entitled to contain. Steps whose columns all lie at or
        below `HDIM_QK_FLOOR` are skipped -- the dispatcher guarantees they are
        real data -- which is what keeps a padded build off the 27-54% the
        forward measured for masking every step.
        """
        traits = self.traits
        steps = k_steps16(traits)
        packs = [
            [self.load_a_tokens(lane, s, ks) for ks in range_constexpr(steps)]
            for s in (0, 1)
        ]
        if const_expr(not self.ctx.PADDED_HEAD):
            return packs
        masked = [ks for ks in range(steps) if (ks + 1) * MFMA16_K > self.hdim_floor]
        if const_expr(not masked):
            return packs
        cols = fmha.MaskedAxis(
            fx.Index(self.hdim),
            elem_dtype=self.ctx.elem_dtype,
            bitmask=len(masked) * (LANE_ELEMS // 2) <= 16,
        )
        for ks in masked:
            col0 = LANE_ELEMS * (lane // fx.Index(MFMA16_M)) + fx.Index(ks * MFMA16_K)
            for st in (0, 1):
                packs[st][ks] = cols.discard(packs[st][ks], col0, LANE_ELEMS)
        return packs

    # -- GEMM3: A operand with the head dim on M, through the transpose -----

    def _kt_read_base(self, lane):
        """Per-lane base for the transposed A operand, in the accumulator's token order.

        The published `a16_read_base` puts `quad * (lane // 16)` tokens in the
        group term, which delivers `k = 8g + i`. This body needs
        `k = 4g + 16r + i`, because that is the order the score accumulator
        already holds (see the module docstring). Both terms are separable and
        `tok_off` is additive across them here -- `(4g + c) // 4 == g` and
        `(4g + c) % 4 == c` for `c < 4 == SMEM_N_RPT` -- so the change is the
        group multiplier and nothing else.
        """
        traits = self.traits
        g = lane // fx.Index(MFMA16_M)
        c = (lane % fx.Index(MFMA16_M)) // fx.Index(4)
        return (
            g * fx.Index(traits.D_128B_SIZE)  # tok_off(4 * g), with SMEM_N_RPT == 4
            + c * fx.Index(self.line)  # tok_off(c)
            + (lane % fx.Index(4)) * fx.Index(4)  # the four d columns this lane fetches
        )

    def load_kt(self, lane):
        """`d_chunks16` transposed A packs, one per 16-wide output chunk."""
        traits = self.traits
        base = fx.Index(self.base_elems) + self._kt_read_base(lane)
        # `+16` tokens, which is `MFMA16_M`, not `2 * MFMA16_M`: the required
        # order is `T(g, 4r + i) = 4g + 16r + i`, so read `r = 1` is one M-step
        # of tokens further on. Getting this wrong reads past the tile and
        # returns finite garbage -- it did.
        pair_bytes = self._tok_off(MFMA16_M) * traits.BF16_BYTES
        packs = []
        for c in range_constexpr(d_chunks16(traits)):
            imm = self._d_off(c * MFMA16_M) * traits.BF16_BYTES

            def read(off):
                return _ds_read_tr_v4f16_imm(
                    base,
                    off,
                    lds_kv_base_idx=self.ctx.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.ctx.v_lds_read_vec4_type,
                    scope_name=self.scope,
                    scope_names=traits.LDS_SCOPE_NAMES,
                )

            lo, hi = read(imm), read(imm + pair_bytes)
            packs.append(Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class M16QLoader:
    """Q or dO as a B operand at 16 rows.

    `B[k][n]`: lane holds `n = q row = lane % 16` and the eight head-dim
    columns `8 * (lane // 16) + i`, so one 128-bit `buffer_load` per k-step --
    the same shape as the 32-row loader, with both lane terms re-based from 32
    to 16.
    """

    def __init__(self, ctx, div, stride_seq_v, gmem_elem_offset, hdim):
        self.ctx = ctx
        self.traits = ctx.traits
        self.div = div
        self.stride = stride_seq_v
        self.gmem = gmem_elem_offset
        self.hdim = hdim

    def load_all(self, q_row_in_block, lane):
        traits = self.traits
        packs = []
        for ks in range_constexpr(k_steps16(traits)):
            col = LANE_ELEMS * (lane // fx.Index(MFMA16_M)) + fx.Index(ks * MFMA16_K)
            raw = dualwave._buffer_load_128(
                self.gmem + q_row_in_block * self.stride + col,
                _load_atom_128=self.ctx.load_atom_128,
                q_div=self.div,
                q_load_i32x4_type=self.ctx.q_load_i32x4_type,
            )
            pack = Vec(raw, (4,), fx.Int32).bitcast(self.ctx.elem_dtype).ir_value()
            if const_expr(self.ctx.PADDED_HEAD):
                # Same argument as the 32-row loader's: a zero in the reduction
                # operand annihilates whatever the other side holds at that
                # column, and the pad's contents are the caller's business.
                pack = fmha.MaskedAxis(
                    fx.Index(self.hdim), elem_dtype=self.ctx.elem_dtype, bitmask=True
                ).discard(pack, col, LANE_ELEMS)
            packs.append(pack)
        return packs


class M16DqGemmHelper:
    """The three GEMMs, on `v_mfma_f32_16x16x32_bf16`."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.atom = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA16_M, MFMA16_M, MFMA16_K, ctx.elem_dtype)
        )
        self.acc_ty = Vec.make_type(acc_elems(ctx.traits), fx.Float32)
        self.zero = Vec.filled(acc_elems(ctx.traits), 0.0, fx.Float32)

    def _mfma(self, a, b, c):
        return dualwave._mfma_acc(a, b, c, self.atom, self.acc_ty)

    def qk(self, a_packs, b_packs):
        """`S = A . B^T` over the head dim, for both KV M-steps.

        GEMM1 with `(K, Q)` and GEMM2 with `(V, dO)` are the same call: the
        operands differ, the contraction does not.
        """
        out = []
        for step in (0, 1):
            acc = self.zero
            for ks in range_constexpr(k_steps16(self.traits)):
                acc = self._mfma(a_packs[step][ks], b_packs[ks], acc)
            out.append(acc)
        # The head_dim 96 workaround the 32-row family carries. Same producer
        # (the QK burst) and same near-zero cost; kept because the underlying
        # defect is latent rather than fixed.
        dualwave._s_nop(1)
        return tuple(out)

    def pv(self, ds_pack, kt_packs, v_dq):
        """`dQ += K^T . dS`, one MFMA per 16-wide output chunk.

        One k-step: `BLOCK_N` is 32 and the MFMA contracts 32, so the whole KV
        tile is a single step -- where the 32-row family needs four substeps.
        """
        for c in range_constexpr(d_chunks16(self.traits)):
            v_dq[c] = self._mfma(kt_packs[c], ds_pack, v_dq[c])
        return v_dq


class M16DqSoftmax:
    """`P`, `dS` and the KV tail mask over a 4-element score half.

    Every method here is a `flash_attn_utils` helper re-spelled against
    `ACC_ELEMS`; that file hardcodes 16 and is production code that must not be
    edited. They are short because the pair structure is preserved -- only the
    element count changed.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.n = acc_elems(ctx.traits)
        score_msteps(ctx.traits)  # the (s_lo, s_hi) precondition, checked once

    def to_lists(self, v_s):
        return (
            [Vec(v_s[0])[r] for r in range_constexpr(self.n)],
            [Vec(v_s[1])[r] for r in range_constexpr(self.n)],
        )

    def to_vecs(self, lists):
        return tuple(
            Vec.from_elements([as_mlir_value(v) for v in half], fx.Float32).ir_value()
            for half in lists
        )

    def scale_and_sub_lse(self, v_s, qk_scale, lse2):
        """`qk_scale * S - lse2`, one FMA per element.

        B2's finding, unchanged: folding `qk_scale` into Q and rounding back to
        bf16 puts `|S| * 2^-8` into the *exponent*, which `dQ` does not
        normalise away. Measured at 10.9x the error ratio at `sm_scale` 1.0.
        """
        fm = self.ctx.fm_fast
        scale_v = Vec.from_elements([fx.Float32(qk_scale)], fx.Float32).broadcast_to(
            self.n
        )
        neg = _fsub(self.ctx.c_zero_f, lse2, fm)
        neg_v = Vec.from_elements([fx.Float32(neg)], fx.Float32).broadcast_to(self.n)
        return tuple(
            as_mlir_value(fmath.fma(Vec(h), scale_v, neg_v, fastmath=fm)) for h in v_s
        )

    def kv_col(self, tile_idx, lane, step, i):
        """The KV column of score element `(step, i)`.

        A lane's four accumulator elements are four **contiguous** tokens at
        `4 * (lane // 16)` -- the 16-row map from `fmha_mfma16_gfx950`, and
        much simpler than the 32-row one, whose four runs of four are why
        `_score_column_runs` exists. The KV tail mask and the dB store both
        read this, so they cannot disagree.
        """
        return (
            fx.Int32(tile_idx * self.traits.BLOCK_N)
            + fx.Int32(step * MFMA16_M + i)
            + fx.Int32(4 * (lane // fx.Index(MFMA16_M)))
        )

    def add_bias(self, v_s, tile_idx, lane, q_row, lse, sm_scale):
        """The bias build's exponent `(fma(S, sm_scale, bias) - lse) * log2e`, formed at the bias add, from this family's own column map.

        A lane's four accumulator elements are four **contiguous** KV columns
        at `4 * (lane // 16)`, so each half is a single 4-wide `buffer_load`.
        The 32-row path needs `_score_column_runs` because its sixteen are four
        scattered spans; reusing that table here would be a transcription.

        **Not `fm_fast`.** That carries `ninf`, a licence to assume no
        infinities reach the operation -- and a bias entry of `-inf` is how a
        caller spells "never attend here", so this is the first thing in the
        kernel that puts a real infinity into *arithmetic* rather than into a
        select. The add takes `FM_BIAS_EXP`.

        The descriptor is bounded at the (batch, head) slab, so a read past the
        last row returns zero rather than faulting -- which is also the right
        bias for a padded row -- and a column past `seqlen_kv` picks up the
        next row's entry, which the KV tail mask below then overwrites with
        `-inf`. That is the forward's argument, unchanged.

        **2-byte reads, one column at a time, not one `dwordx2` per run.**
        gfx950 range-checks a multi-dword buffer op per dword and drops any
        dword not wholly inside `num_records`, and the slab ends exactly on the
        last valid element -- so a wide read of the last row with odd
        `seqlen_kv` loses column `seqlen_kv-1` to the out-of-range column
        `seqlen_kv` sharing its dword. The forward narrows only on its
        tail-masked tiles; here every tile takes the mask (see the call at the
        `M16TileBody` loop), so there is no wide case to keep.
        """
        ctx = self.ctx
        row_base = q_row * fx.Index(ctx.stride_b_seq_q)
        lists = self.to_lists(v_s)
        log2e = fx.Float32(dualwave._LOG2E)
        for step in range_constexpr(2):
            col0 = self.kv_col(tile_idx, lane, step, 0)
            for i in range_constexpr(self.n):
                one = buffer_ops.buffer_load(
                    ctx.bias_rsrc,
                    as_mlir_value(fx.Int32(row_base + fx.Index(col0 + i))),
                    vec_width=1,
                    dtype=ctx.elem_dtype,
                )
                # `vec_width=1` returns a **scalar**, not a one-lane vector.
                b = fx.Float32(ctx.elem_dtype(one).to(fx.Float32))
                lists[step][i] = _fmul(
                    _fsub(
                        fmath.fma(lists[step][i], sm_scale, b, fastmath=FM_BIAS_EXP),
                        lse,
                        FM_BIAS_EXP,
                    ),
                    log2e,
                    FM_BIAS_EXP,
                )
        return self.to_vecs(lists)

    def seq_pad_mask(self, v_s, tile_idx, lane):
        """`-inf` past `seqlen_kv`, on the tiles that need it.

        Guarded by a traced `if` rather than applied to every tile: the
        forward measured 27-54% for an unconditional per-step mask, and this
        one is only ever live on the last tile.
        """
        traits = self.traits
        seqlen_kv_i32 = self.ctx.seqlen_kv_i32
        neg_inf = self.ctx.c_neg_inf
        to_lists, to_vecs, kv_col, n = self.to_lists, self.to_vecs, self.kv_col, self.n

        @flyc.jit
        def _mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            if fx.Int32((tile_idx + fx.Index(1)) * traits.BLOCK_N) > seqlen_kv_i32:
                lists = to_lists(v_s)
                for step in range_constexpr(2):
                    for i in range_constexpr(n):
                        live = kv_col(tile_idx, lane, step, i) < seqlen_kv_i32
                        lists[step][i] = live.select(
                            fx.Float32(lists[step][i]), neg_inf
                        )
                s_lo, s_hi = to_vecs(lists)
            return s_lo, s_hi

        return _mask_if_needed(v_s, tile_idx)

    def causal_mask(self, v_s, tile_idx, lane, q_row):
        """The causal right bound, and a window's left one, at 16 rows.

        **Derived from this family's own lane map**, not transcribed from the
        32-row one: `kv_col` is the single place the element -> column map
        lives here, and the KV tail mask and the dB store read the same
        function. The 32-row path cannot be reused at all -- its
        `_causal_pair_thresholds` table describes a 16-element accumulator half
        whose columns are four scattered runs, where this one holds four
        contiguous tokens.

        The guard is the forward's two-sided test, over the *wave's* row range
        rather than the workgroup's, because a wave owns 16 rows here:

        - a column can overrun the right bound only if the **lowest** row's
          bound lands inside the tile;
        - a column can fall behind the left bound only if the **highest** row's
          edge is still above `kv_start`.

        Both terms are wave-uniform (`wave_id_uni` is `readfirstlane`d), so the
        branch is scalar and **EXEC is untouched** -- which is what keeps the
        transpose read below it legal under CDNA4 11.4. A divergent condition
        here would be a correctness bug, not a slow one.

        Plain `select`s rather than the 32-row path's paired inline asm: that
        exists because the causal mask is on the innermost path of every
        forward build, and reaching for it here before a measurement asks would
        be copying a decision rather than its reason.
        """
        traits = self.traits
        ctx = self.ctx
        neg_inf = ctx.c_neg_inf
        delta_i32 = ctx.delta_i32
        to_lists, to_vecs, kv_col, n = self.to_lists, self.to_vecs, self.kv_col, self.n
        windowed = const_expr(traits.WINDOW)
        window_left_i32 = ctx.window_left_i32 if windowed else None
        # The wave's first row, uniform. `q_row` is this lane's and is not.
        qs_i32 = fx.Int32(ctx.q_start) + fx.Int32(ctx.wave_id_uni) * fx.Int32(MFMA16_M)
        q_row_i32 = fx.Int32(q_row)

        @flyc.jit
        def _mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_end = fx.Int32((tile_idx + fx.Index(1)) * traits.BLOCK_N)
            need = qs_i32 + delta_i32 < kv_end
            if windowed:
                kv_start = fx.Int32(tile_idx * traits.BLOCK_N)
                need = need | (
                    kv_start < qs_i32 + fx.Int32(MFMA16_M - 1) - window_left_i32
                )
            if need:
                lists = to_lists(v_s)
                for step in range_constexpr(2):
                    for i in range_constexpr(n):
                        col = kv_col(tile_idx, lane, step, i)
                        keep = col <= q_row_i32 + delta_i32
                        if windowed:
                            keep = keep & (col >= q_row_i32 - window_left_i32)
                        lists[step][i] = keep.select(
                            fx.Float32(lists[step][i]), neg_inf
                        )
                s_lo, s_hi = to_vecs(lists)
            return s_lo, s_hi

        return _mask_if_needed(v_s, tile_idx)

    def exp2(self, v_s):
        """The `exp2` batch, held one slot away from its consumers.

        `exp2_wait_state` carries the argument; the 32-row family's `exp2`
        carries the measurement for dQ, which is that no dQ build scanned comes
        within 8 instructions of the hazard. Imported as a property of the
        instruction pair rather than of today's schedule.

        Unlike the 32-row path this one returns the lists directly, so it wraps
        rather than needing the body copied.
        """
        return tuple(
            exp2_wait_state(
                [
                    dualwave.rocdl.exp2(T.f32, as_mlir_value(Vec(h)[r]))
                    for r in range_constexpr(self.n)
                ]
            )
            for h in v_s
        )

    def dropout_dp(self, dp_lists, tile_idx, lane, q_row):
        """`dP <- keep ? dP * (1/(1-p)) : 0`, at 16 rows.

        Same placement and same reasoning as the 32-row family's
        `BwdDqSoftmaxHelper.dropout_dp` -- the mask is on the gradient of the
        dropped output, `P` stays undropped, and `delta` already carries the
        factor through the forward's `O`.

        The span is simpler here and that is the family's own map, not a
        transcription: a lane's four accumulator elements are four
        **contiguous** KV columns starting at `4 * (lane // 16)`, which is a
        multiple of `randoms_per_offset`, so each half is exactly one Philox
        call with no partial draw. The 32-row path needs
        `_score_column_runs` because its four are scattered.
        """
        ctx = self.ctx
        rng = ctx.philox_rng
        scale = fx.Float32(ctx.dropout_scale_arg)
        zero = ctx.c_zero_f
        for step in range_constexpr(2):
            first = rng.grid_offset(
                ctx.philox_plane_base,
                ctx.philox_row_stride,
                fx.Int64(q_row),
                fx.Int64(self.kv_col(tile_idx, lane, step, 0)),
            )
            keep = rng.keep_span(ctx.philox_seed, first, self.n, ctx.idropout_p)
            for i in range_constexpr(self.n):
                kept = _fmul(dp_lists[step][i], scale, ctx.fm_fast)
                dp_lists[step][i] = keep[i].select(fx.Float32(kept), zero)
        return dp_lists

    def dscores(self, p_lists, dp, delta):
        """`dS = P * (dP - delta)`, elementwise. `dp` is already a list pair."""
        fm = self.ctx.fm_fast
        return tuple(
            [
                _fmul(p_lists[h][r], _fsub(dp[h][r], delta, fm), fm)
                for r in range_constexpr(self.n)
            ]
            for h in (0, 1)
        )

    def pack_ds(self, ds_lists):
        """One v8 bf16 B operand out of the two 4-element halves.

        `[lo0..lo3, hi0..hi3]`, in that order, because slot `4r + i` of the
        operand must hold the token the transpose read put at `4g + 16r + i` --
        which is accumulator half `r`, element `i`. The whole permutation
        argument reduces to this concatenation being in the obvious order.

        The pack leaves through `mfma_operand_wait_state` for the same reason
        dK/dV's does; that docstring has the measurement. dQ has not been seen
        to fail this way, but it builds the operand with the same instruction
        and hands it to the same MFMA shape, and the barrier is free.
        """
        vals = list(ds_lists[0]) + list(ds_lists[1])
        return mfma_operand_wait_state(
            dualwave._bf16_trunc_pack_v8(
                self.traits, vals, elem_dtype=self.ctx.elem_dtype
            )
        )


class M16DqStore:
    """dQ, four contiguous columns per lane per chunk.

    **No cross-lane shuffle.** The 32-row family's O store runs the
    accumulator through `permlane32_swap` (`_fused_o_128_dwords`) because a
    lane's 16 values are scattered `[d]` for one `[q]` and a row-major store
    needs them gathered. At 16 rows a lane's four accumulator elements are four
    *contiguous* d columns of one query row, so the store is one 8-byte write.

    Suppression past `hdim_qk` is by address, as everywhere else here: pushing
    the offset past the descriptor's `num_records` makes the hardware drop it.
    A run may write up to three columns into the caller's pad, which the 8xD
    contract guarantees exists -- and is stricter than the 32-row path, whose
    128-bit store can overrun by eight.

    The descriptor is bounded at the last row's last real element rather than
    at `rows * stride`, which is what makes that pad claim true for the *last*
    (batch, head) slab of a BSHD dQ, where the row stride is `num_heads * hdim`
    and the pad it would otherwise write into is off the end of the tensor.
    See `_slab_span_elems`. `oob` stays the untightened span: a sentinel only
    has to sit at or past `num_records`, and shrinking the bound moves it
    further out of range rather than back into it.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.oob = ctx.dq_oob_off
        # `hdim_qk`, not `hdim_vo`: dQ is Q-shaped, as `store` says again.
        span = _slab_span_elems(ctx.seqlen_q_v, ctx.stride_dq_seq, ctx.hdim_qk)
        self.rsrc = buffer_ops.create_buffer_resource(
            ctx.DQ,
            max_size=False,
            num_records_bytes=as_mlir_value(span * fx.Index(self.traits.BF16_BYTES)),
            base_byte_offset=as_mlir_value(
                ctx._slab_byte_base(
                    ctx.stride_dq_batch,
                    ctx.stride_dq_head,
                    ctx.stride_dq_seq,
                    ctx.q_row_off,
                    ctx.q_head_idx,
                )
            ),
        )

    def store(self, v_dq, q_row, lane):
        ctx, traits = self.ctx, self.traits
        n = acc_elems(traits)
        row_base = q_row * ctx.stride_dq_seq_v
        in_row = q_row < ctx.seqlen_q_v
        col_lane = fx.Index(4) * (lane // fx.Index(MFMA16_M))
        for c in range_constexpr(d_chunks16(traits)):
            col = fx.Index(c * MFMA16_M) + col_lane
            vals = [
                fx.Float32(Vec(v_dq[c])[r]).to(ctx.elem_dtype)
                for r in range_constexpr(n)
            ]
            pack = Vec.from_elements(vals, ctx.elem_dtype)
            live = in_row
            if const_expr(ctx.PADDED_HEAD):
                # `hdim_qk`, not `hdim_vo`: dQ is Q-shaped. The two extents are
                # used the other way round in this kernel than in the forward.
                live = live & (col < fx.Index(ctx.hdim_qk))
            off = fx.Index(live.select(row_base + col, self.oob))
            buffer_ops.buffer_store(
                as_mlir_value(pack), self.rsrc, as_mlir_value(fx.Int32(off))
            )


def make_m16_dq_body(
    ctx,
    traits,
    *,
    kv_gmem_to_lds,
    db_store,
    hdim_qk,
    hdim_vo,
    hdim_qk_floor,
    hdim_vo_floor,
    store_db,
):
    """The traced 16-row body. A factory, because of the AST rewrite:

    The `for ... init=[...]` / `yield` protocol only exists after the AST
    rewrite, so a module-level function called from the kernel is compiled by
    Python as written and `range(..., init=...)` is a `TypeError`. Closing over
    the helpers keeps them out of the traced signature, where only values
    belong.
    """
    lane = ctx.lane
    # 16 rows per wave, so the wave's row block is 16 wide and the lane's row
    # is `lane % 16` -- `_init_dualwave_q_row`'s `lane_mod_32` is the 32-row
    # spelling of this and is not reusable.
    q_row_in_block = ctx.wave_id * fx.Index(MFMA16_M) + lane % fx.Index(MFMA16_M)
    q_row = ctx.q_start + q_row_in_block

    k_reader = M16DqReader(
        ctx,
        region="v",
        line_stride=traits.SMEM_V_LINE_STRIDE,
        hdim=hdim_qk,
        hdim_floor=hdim_qk_floor,
    )
    v_reader = M16DqReader(
        ctx,
        region="k",
        line_stride=traits.SMEM_K_LINE_STRIDE,
        hdim=hdim_vo,
        hdim_floor=hdim_vo_floor,
    )
    q_loader = M16QLoader(
        ctx, ctx.q_div, ctx.stride_q_seq_v, ctx.q_gmem_elem_offset, hdim_qk
    )
    do_loader = M16QLoader(
        ctx, ctx.do_div, ctx.stride_do_seq_v, ctx.do_gmem_elem_offset, hdim_vo
    )
    gemm = M16DqGemmHelper(ctx)
    softmax = M16DqSoftmax(ctx)
    store = M16DqStore(ctx)
    n_chunks = d_chunks16(traits)

    @flyc.jit
    def _m16_body():
        """One KV tile of 32 per iteration; three GEMMs and one accumulator."""
        q_packs = q_loader.load_all(q_row_in_block, lane)
        do_packs = do_loader.load_all(q_row_in_block, lane)
        lse2, delta = ctx.load_row_scalars(q_row)

        init_args = [gemm.zero for _ in range_constexpr(n_chunks)]
        loop_results = init_args
        for j, loop_args in range(
            ctx.split_tile(0), ctx.split_t_end, fx.Index(1), init=init_args
        ):
            v_dq = [loop_args[i] for i in range_constexpr(n_chunks)]
            tile_start = ctx.tile_start(j)

            _s_barrier()
            kv_gmem_to_lds.stage_k(tile_start)
            kv_gmem_to_lds.stage_v(tile_start)
            _s_waitcnt(0)
            _sched_barrier(0)
            _s_barrier()

            v_s = gemm.qk(k_reader.load_a_all(lane), q_packs)
            if const_expr(traits.BIAS_TYPE):
                # Natural units, `(fma(S, sm_scale, bias) - lse) * log2e`, formed at the bias add, before the masks.
                v_s = softmax.add_bias(v_s, j, lane, q_row, lse2, ctx.c_sm_scale)
            else:
                v_s = softmax.scale_and_sub_lse(v_s, ctx.c_sm_scale_log2e, lse2)
            v_s = softmax.seq_pad_mask(v_s, j, lane)
            if const_expr(traits.CAUSAL):
                # Same placement argument as the 32-row body's: this touches
                # `v_s` only, and every transpose read is below it. The guard
                # is wave-uniform, so the branch is scalar and EXEC stays all
                # ones across `load_kt`.
                v_s = softmax.causal_mask(v_s, j, lane, q_row)
            v_dp = gemm.qk(v_reader.load_a_all(lane), do_packs)

            dp_lists = softmax.to_lists(v_dp)
            if const_expr(traits.ENABLE_DROPOUT):
                dp_lists = softmax.dropout_dp(dp_lists, j, lane, q_row)
            ds = softmax.dscores(softmax.exp2(v_s), dp_lists, delta)
            ds_pack = softmax.pack_ds(ds)
            if const_expr(store_db):
                db_store.store_tile_m16(ds_pack, j, q_row, lane)
            v_dq = gemm.pv(ds_pack, k_reader.load_kt(lane), v_dq)

            loop_results = yield v_dq

        v_dq = [loop_results[i] for i in range_constexpr(n_chunks)]
        scale = Vec.from_elements([ctx.c_sm_scale], fx.Float32).broadcast_to(
            acc_elems(traits)
        )
        for c in range_constexpr(n_chunks):
            v_dq[c] = _fmul(Vec(v_dq[c]), scale, ctx.fm_fast)
        _s_barrier()
        store.store(v_dq, q_row, lane)

    return _m16_body
