# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""The dK/dV tile body at **16 KV rows per wave**, on `v_mfma_f32_16x16x32`.

A second family alongside the 32-row one in `fmha_bwd_dkdv_gfx950.py`,
not a replacement: the tuning table picks per rung. A separate module because
the operand algebra differs at a dozen sites, so an `if` at each would be a
fork with extra steps. Everything that is
not operand algebra stays shared: the context, the DMA staging, the
descriptors, the ABI and the tuning.

--- Why 16 rows -------------------------------------------------------------

A dK/dV wave holds two accumulators and its resident K and V. At 32 rows that
is `1.5 * d` VGPRs before a transient -- 768 at head_dim 512 against a file of
512 -- so the 32-row family buys the space with `DKV_SHARDS`, which divides `BLOCK_KV` and
therefore multiplies the Q/dO re-read traffic *and* recomputes S and dP per
shard. At 16 rows it is `0.75 * d`: 384 at head_dim 512, **unsharded**.

    per wave, VGPRs      32 rows        16 rows
    resident K + V       d/4 + d/4      d/8 + d/8
    dK acc + dV acc      d/2 + d/2      d/4 + d/4
    total                1.5 d          0.75 d

--- Why `16x16x32` and not `16x16x16` ---------------------------------------

**`16x16x16` is half rate**, which makes a family built on it unable to beat
the 32-row one anywhere the 32-row one fits. From `SISchedule.td`'s
`SIDPGFX950FullSpeedModel`, where `16x16x16` and `16x16x32` are both
`Write4PassMAI` and `32x32x16` is `Write8PassMAI`:

| shape | FLOPs | passes | FLOPs/pass |
|---|---|---|---|
| `16x16x16` | 8192 | 4 | **2048** |
| `16x16x32` | 16384 | 4 | 4096 |
| `32x32x16` | 32768 | 8 | 4096 |

This module was first built on `16x16x16` and measured 280 TFLOP/s at head_dim
64 against the 32-row family's 713 -- half the rate and half the `BLOCK_KV`,
which is exactly those two factors. `16x16x32` keeps 16 rows at full rate, and
it keeps the operand eight elements wide, so `_bf16_trunc_pack_v8` and the
128-bit LDS reads survive unchanged.

The transpose read serves this shape's A operand end to end at both staging
granules, provided a staged tile has at least four lines (`SMEM_N_RPT >= 4`),
which `BwdDkDvKnobs` enforces and `init16` asserts.

--- The one thing that does not line up, and the free fix -------------------

The `16x16x32` B operand wants eight contraction values per lane,
`k = 8*(lane//16) + i`, while a `16x16` accumulator holds four,
`m = 4*(lane//16) + i`. So `P` for one 32-row q group has to come from **two**
score accumulators -- and the naive pairing does not match: lane group `g`
needs `q = 8g..8g+7`, while sub-blocks 0 and 1 offer it `4g..4g+3` and
`16+4g..16+4g+3`. Half the values live in a *different quarter-wave*.

The fix costs nothing, because **which q row lands on which accumulator row is
ours to choose**: the score GEMM's A operand is the staged Q tile, read at a
per-lane row index. Feeding sub-block `s` the row permutation

    q(m) = 8*(m//4) + (m%4) + 4*s          m = lane % 16

makes the accumulator hold `q = 8*(lane//16) + i + 4*s`, so concatenating the
two sub-blocks gives `k = 8*(lane//16) + 0..7` exactly. It is a different
address, not a shuffle, and it is loop-invariant.

It also leaves a lane's four accumulator rows **contiguous** (`8*(lane//16) +
4*s + 0..3`), so the LSE and delta reads stay one `dwordx4` each -- where the
32-row family needs `_score_column_runs`' four spans.

--- What else is simpler ----------------------------------------------------

- **The transpose read carries no permutation on the contraction axis.** The
  32-row map's is `[0,1,2,3,8,9,10,11]`; here `k = 8*(lane//16) + i`.
- **The store needs no cross-lane shuffle.** A lane holds four *contiguous* d
  of one KV row -- one 8-byte store -- where the 32-row path assembles 128 bits
  through `permlane32_swap` first.
- One staged tile per tensor, read two ways, as in the 32-row family: the
  transpose serves a 16-row operand from the *same* bytes the row-major read
  uses, so no second LDS copy of the tile is needed.
"""

from kernels.attention_bwd_cdna.fmha_bwd_bias_exp_gfx950 import FM_BIAS_EXP
from kernels.attention_bwd_cdna.fmha_common_gfx1201 import MaskedAxis
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import _fadd, _fmul, _fsub
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    _ds_read_tr16_b64_imm,
    exp2_wait_state,
    mfma_operand_wait_state,
)
from kernels.attention_bwd_cdna.fmha_mfma16_gfx950 import (
    MFMA16_M,
    a16_chunk_offset,
    a16_read_base,
    lds_elem,
    tok_off,
    tok_off_dyn,
)
from kernels.attention_bwd_cdna.gfx950_standalone import buffer_ops, dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

__all__ = [
    "ACC16",
    "M16GemmHelper",
    "M16ResidentLoader",
    "M16SoftmaxHelper",
    "M16StoreHelper",
    "M16StreamReader",
    "M16TileBody",
    "d_chunks16",
    "k_steps32",
    "pack_split",
]

# The MFMA's contraction extent, elements per lane in an A or B operand, and
# f32 per lane in the 16x16 accumulator.
MFMA16_K = 32
LANE_K = MFMA16_K * MFMA16_M // 64  # 8
ACC16 = MFMA16_M * MFMA16_M // 64  # 4


def pack_split(ctx, x):
    """`gqa_pack`: flat row `x` of the packed q axis -> `(head g, q row, valid)`.

    The packed axis lays the GQA group's heads end to end, `x = g * Sq + q`, and one tile walk covers it. `g = x // Sq` is a
    multiply-high by `pack_magic = (2^31 - 1) // Sq + 1`, exact only while `x * Sq < 2^31` for every row the walk addresses, pad included
    (`check_launch_limits` in the tuning module refuses a launch beyond it), so no
    per-element divide is emitted. `valid` is `x < Sq * G`: rows past it are the tile's pad and every address built from them is
    redirected out of its descriptor, so they stage and read as zero exactly as a row past `seqlen_q` does in the unpacked path.
    """
    g = fx.Index((fx.Int64(x) * ctx.pack_magic).shrui(fx.Int64(31)))
    return g, x - g * ctx.pack_sq, x < ctx.pack_rows


def q_groups(traits):
    """32-row q groups in one streamed tile: one `16x16x32` contraction each."""
    return traits.BLOCK_Q // MFMA16_K


def score_subs(traits):
    """16x16 score accumulators in one streamed tile. Two per q group."""
    return traits.BLOCK_Q // MFMA16_M


def k_steps32(traits):
    """`16x16x32` steps to contract the head dim."""
    return traits.HEAD_DIM // MFMA16_K


def d_chunks16(traits):
    """16-wide output chunks one wave owns.

    **This family does not shard.** It does not need to -- `0.75 * d` fits at
    head_dim 512 where the 32-row family's `1.5 * d` does not -- and not
    sharding keeps `a16_chunk_offset` a compile-time immediate, where a runtime
    shard origin would have to be folded into the read base as the 32-row path
    does. `BwdDkDvKnobs` rejects the combination rather than leaving it as a
    silent precondition.
    """
    return traits.HEAD_DIM // MFMA16_M


def _masked_ks_steps(traits, hdim_floor):
    """The `16x16x32` k-steps whose D columns can reach past the real head dim."""
    return [ks for ks in range(k_steps32(traits)) if (ks + 1) * MFMA16_K > hdim_floor]


def _lds_pack_read(traits, lds_ptr, elem_idx, scope_name, pack_type):
    """One 128-bit LDS read of 8 bf16, alias-scoped to the buffer it touches.

    `dualwave._load_k_pack_aligned` with the scope name as a parameter; it
    derives its own as `lds_k{buf}`, which is wrong here where two tensors are
    staged and the second lives in the `v` scopes.
    """
    ptr = buffer_ops.get_element_ptr(
        lds_ptr, byte_offset=elem_idx * traits.BF16_BYTES, elem_type=T.i8
    )
    return dualwave.llvm.LoadOp(
        pack_type,
        ptr,
        alignment=16,
        alias_scopes=dualwave._dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=dualwave._dualwave_lds_noalias_scopes(
            scope_name, traits.LDS_SCOPE_NAMES
        ),
    ).result


class M16StreamReader(dualwave.DualwaveKernelContext):
    """The two ways one staged tile is read back, at 16 rows.

    Row-major for the d-contracted GEMMs and column-major through
    `ds_read_b64_tr_b16` for the q-contracted ones, from the *same* LDS bytes.
    """

    masked_steps = ()

    def init16(self, hdim_floor):
        """Per-lane bases, computed once. All loop-invariant across q tiles."""
        traits = self.traits
        lane = self.lane
        assert (
            traits.SMEM_N_RPT >= 4
        ), f"the 16-row transposed read addresses four staged lines per q group; SMEM_N_RPT is {traits.SMEM_N_RPT}"
        self.masked_steps = tuple(_masked_ks_steps(traits, hdim_floor))
        # Row-major, per score sub-block: the permuted q row (see the module
        # docstring), plus eight contiguous d at `8 * (lane // 16)`. `tok_off`
        # is mixed-radix, so the row term goes through the dynamic form.
        lm = lane % fx.Index(MFMA16_M)
        perm_row = fx.Index(8) * (lm // fx.Index(4)) + (lm % fx.Index(4))
        self.row_base16 = [
            tok_off_dyn(
                traits, fx.Index((sub // 2) * MFMA16_K + (sub % 2) * 4) + perm_row
            )
            + (lane // fx.Index(MFMA16_M)) * fx.Index(LANE_K)
            for sub in range(score_subs(traits))
        ]
        # Column-major: the published A-operand base, `quad = 8` for K=32.
        self.col_base16 = a16_read_base(traits, lane, LANE_K)

    def read_row_pack(self, tile_base, scope_name, sub, ks, hdim):
        """`A[q][d]`: eight d of this lane's permuted q row, for k-step `ks`."""
        traits = self.traits
        idx = tile_base + self.row_base16[sub] + lds_elem(traits, 0, ks * MFMA16_K)
        pack = _lds_pack_read(
            traits, self.lds_kv_base_ptr, idx, scope_name, self.kv_mfma_pack_type
        )
        if const_expr(self.PADDED_HEAD) and ks in self.masked_steps:
            col = fx.Index(ks * MFMA16_K) + (
                self.lane // fx.Index(MFMA16_M)
            ) * fx.Index(LANE_K)
            pack = MaskedAxis(
                fx.Index(hdim), elem_dtype=self.elem_dtype, bitmask=True
            ).discard(pack, col, LANE_K)
        return pack

    def read_column_chunk(self, tile_base, scope_name, c):
        """`A[d][q]` for output chunk `c`, one 8-element pack per 32-row q group.

        Two `ds_read_b64_tr_b16` each, the second four tokens on -- the same
        pair structure the 32-row family uses, with the pair stride at 4 tokens
        rather than 8 because a lane group here spans 8 tokens rather than 16.
        """
        traits = self.traits
        addr = fx.Int32(
            (tile_base + self.col_base16) * traits.BF16_BYTES + self.lds_kv_base_idx
        )
        chunk_off = a16_chunk_offset(traits, c) * traits.BF16_BYTES
        pair = tok_off(traits, 4) * traits.BF16_BYTES
        packs = []
        for g in range_constexpr(q_groups(traits)):
            base = chunk_off + tok_off(traits, g * MFMA16_K) * traits.BF16_BYTES
            a = _ds_read_tr16_b64_imm(
                self.v_lds_read_vec4_type,
                addr,
                base,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            b = _ds_read_tr16_b64_imm(
                self.v_lds_read_vec4_type,
                addr,
                base + pair,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            packs.append(Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class M16ResidentLoader(dualwave.DualwaveKernelContext):
    """K and V rows for this wave, loaded once and held for the whole q loop.

    `B[d][kv]`: `n = kv_row = lane % 16`, `k = d = ks*32 + 8*(lane//16) + i`.
    Eight elements as in the 32-row family, but `head_dim / 32` steps rather
    than `/ 16` -- which is the whole of the resident-register halving.

    A raw bounded resource rather than the copy-atom `_buffer_load_128` path,
    because that atom is built against the 32-row reader's descriptor. The
    bound is identical -- `seqlen_kv * stride_seq` elements -- so a KV row past
    the sequence still reads zero and the ragged tail still needs no mask.
    """

    def load_resident(self, which, hdim):
        traits = self.traits
        rsrc = self.k_res_rsrc if which == "k" else self.v_res_rsrc
        elem_base = self.k_res_elem_base if which == "k" else self.v_res_elem_base
        stride_seq = self.stride_k_seq_v if which == "k" else self.stride_v_seq_v
        row = elem_base + self.kv_row_in_block * stride_seq
        lane_d = (self.lane // fx.Index(MFMA16_M)) * fx.Index(LANE_K)
        cols = (
            MaskedAxis(fx.Index(hdim), elem_dtype=self.elem_dtype, bitmask=True)
            if self.PADDED_HEAD
            else None
        )
        packs = []
        for ks in range_constexpr(k_steps32(traits)):
            col = fx.Index(ks * MFMA16_K) + lane_d
            pack = buffer_ops.buffer_load(
                rsrc,
                as_mlir_value(fx.Int32(row + col)),
                vec_width=LANE_K,
                dtype=self.elem_dtype,
            )
            if const_expr(self.PADDED_HEAD):
                pack = cols.discard(pack, col, LANE_K)
            packs.append(pack)
        return packs


class M16GemmHelper(dualwave.DualwaveKernelContext):
    """The four GEMMs, as two shapes. `16x16x32` throughout."""

    def init16(self):
        self.mma16 = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA16_M, MFMA16_M, MFMA16_K, self.elem_dtype)
        )
        self.acc16_ty = Vec.make_type(ACC16, fx.Float32)
        self.zero16 = Vec.filled(ACC16, 0.0, fx.Float32).ir_value()

    def _mfma(self, a, b, c):
        return dualwave.fly.mma_atom_call_ssa([self.acc16_ty], self.mma16, a, b, c)

    def contract_d(self, read_a, b_packs):
        """`S` or `dP` for one 16-row score sub-block: reduce over the head dim.

        `read_a` is a callable so each A pack is read into the MFMA that
        consumes it rather than the whole set being gathered first.
        """
        acc = self.zero16
        for ks in range_constexpr(k_steps32(self.traits)):
            acc = self._mfma(read_a(ks), b_packs[ks], acc)
        return acc

    def contract_q(self, a_packs, b_packs, acc):
        """`dV` or `dK`: reduce over the tile's 32-row q groups."""
        for g in range_constexpr(q_groups(self.traits)):
            acc = self._mfma(a_packs[g], b_packs[g], acc)
        return acc


class M16SoftmaxHelper(dualwave.DualwaveKernelContext):
    """`P` and `dS` for one 16x16 score tile.

    Four elements per lane rather than thirty-two, and the q row a lane's
    element `i` holds is `8*(lane//16) + 4*s + i` -- **contiguous**, so one
    `dwordx4` covers the LSE and one the delta. The 32-row family needs
    `_score_column_runs`' four spans for the same job.
    """

    def load_row_values(self, rsrc, tile_base, sub, scale):
        """Four f32 from a compact row tensor, times `scale`.

        **Chosen at runtime on `pitch != 1`**, not at build time on
        `lse_layout_th`; `BwdDkDvSoftmaxHelper.load_row_values` carries the
        measurement and the reason. The short version is that
        `lse_row_addressing` decodes the layout from `VarlenBits` at runtime,
        so a build compiled for one layout and handed the other read the wrong
        elements with only a host-side wrapper standing in the way -- and that
        wrapper is not on AOTriton's path.

        Under `_TH` the four rows are `num_heads` apart, so the `dwordx4` does
        not apply and it is four scalars. `LSE_STRIDED` is a Python bool: the
        body is traced once per arm and one runtime branch outside the tile
        loop picks a whole body, which is what keeps the wide arm at its
        pre-fix throughput.

        The scalar arm splits the offset the way the 32-row family's does --
        one divergent multiply and `ACC16` uniform ones, rather than `ACC16`
        divergent ones. Four addresses is not where that family's registers go,
        so it buys little here; it is written the same way because the two
        arms are read together and a second spelling of one address is how the
        layout bug got in.
        """
        if const_expr(getattr(self, "GQA_PACK", False)):
            # Packed rows: element i is flat row `base + i`, addressed `g * head_step + q * pitch` and read as a scalar (a run of
            # four flat rows may straddle two heads). `pitch` is the runtime layout decode, so one body serves both layouts.
            base = (
                tile_base
                + fx.Index((sub // 2) * MFMA16_K + (sub % 2) * 4)
                + (self.lane // fx.Index(MFMA16_M)) * fx.Index(8)
            )
            out = []
            for i in range_constexpr(ACC16):
                g, q, valid = pack_split(self, base + fx.Index(i))
                off = fx.Index(
                    valid.select(
                        g * self.lse_head_step_elems + q * self.lse_pitch,
                        self.pack_lse_oob,
                    )
                )
                one = buffer_ops.buffer_load(
                    rsrc, as_mlir_value(fx.Int32(off)), vec_width=1, dtype=fx.Float32
                )
                out.append(_fmul(fx.Float32(one), scale, self.fm_fast))
            return out
        row0 = fx.Int32(
            tile_base + fx.Index((sub // 2) * MFMA16_K + (sub % 2) * 4)
        ) + fx.Int32(self.lane // fx.Index(MFMA16_M)) * fx.Int32(8)
        if const_expr(not self.LSE_STRIDED):
            span = buffer_ops.buffer_load(
                rsrc, as_mlir_value(row0), vec_width=ACC16, dtype=fx.Float32
            )
            vec = Vec(span, (ACC16,), fx.Float32)
            return [_fmul(vec[i], scale, self.fm_fast) for i in range_constexpr(ACC16)]
        pitch = self.lse_pitch
        row_v = fx.Index(row0) * pitch
        out = []
        for i in range_constexpr(ACC16):
            off = row_v + fx.Index(i) * pitch
            one = buffer_ops.buffer_load(
                rsrc, as_mlir_value(fx.Int32(off)), vec_width=1, dtype=fx.Float32
            )
            out.append(_fmul(fx.Float32(one), scale, self.fm_fast))
        return out

    def q_rows(self, tile_idx, sub):
        """The absolute q row each of this sub-block's four elements holds.

        `(sub//2)*32 + (sub%2)*4 + 8*(lane//16) + i` -- the permuted row map of
        this family, stated once and read by `load_row_values`,
        `mask_if_clipped` and the dropout mask. Never taken from the 32-row
        family's table.
        """
        base = (
            tile_idx * fx.Index(self.traits.BLOCK_Q)
            + fx.Index((sub // 2) * MFMA16_K + (sub % 2) * 4)
            + (self.lane // fx.Index(MFMA16_M)) * fx.Index(8)
        )
        return [base + fx.Index(i) for i in range_constexpr(ACC16)]

    def probabilities(self, v_s, lse_term, bias2=None):
        """`P = exp2(qk_scale * S - log2(e) * LSE)`, or with a bias `exp2((sm_scale * S + bias - LSE) * log2(e))`.

        Q is not pre-scaled: folding the scale into Q and
        rounding to bf16 puts the error in the exponent, taking the error ratio
        from 1.29 at `sm_scale = 0.05` to 10.9 at 1.0.

        A bias build combines in natural units and scales last (`bias2` is the natural-unit
        bias, `lse_term` is `LSE`), and its chain takes `FM_BIAS_EXP` (no `ninf`, no `reassoc`)
        so a caller's `-inf` and an empty row's `+inf` LSE reach `exp2`, which returns an exact
        zero for them. The 32-row family's `probabilities` says why at length.
        """
        values = [Vec(v_s)[r] for r in range_constexpr(ACC16)]
        if const_expr(bias2 is not None):
            fm = FM_BIAS_EXP
            sm_scale = self.ctx_ref.c_sm_scale
            log2e = fx.Float32(dualwave._LOG2E)
            return exp2_wait_state(
                [
                    dualwave.rocdl.exp2(
                        T.f32,
                        as_mlir_value(
                            _fmul(
                                _fsub(
                                    fmath.fma(
                                        values[r], sm_scale, bias2[r], fastmath=fm
                                    ),
                                    lse_term[r],
                                    fm,
                                ),
                                log2e,
                                fm,
                            )
                        ),
                    )
                    for r in range_constexpr(ACC16)
                ]
            )
        scale = self.ctx_ref.c_sm_scale_log2e
        fm = self.fm_fast
        scaled = [_fmul(values[r], scale, fm) for r in range_constexpr(ACC16)]
        # The `v_exp_f32` wait state. `exp2_wait_state` says why it is needed
        # and why `dualwave._s_nop(1)` would not
        # supply it -- the short version is that `s_nop` takes no operands, so
        # nothing stopped the scheduler from hoisting a `v_cvt_pk_bf16_f32`
        # above it and back into the slot right after an `exp2`.
        #
        # Found at `block_dmodel=128, mfma_rows=16`, bf16, `BIAS_TYPE=1`, where
        # the schedule that hits it appears once per iteration: dV came out
        # about five times the reference on the odd tile's second q group. The
        # `_s_nop` fixed *that* build, by luck of the allocation it perturbed,
        # and left `BLOCK_DMODEL=192, PADDED_HEAD=False` with two zero-gap
        # sites still in it.
        return exp2_wait_state(
            [
                dualwave.rocdl.exp2(
                    T.f32, as_mlir_value(_fadd(scaled[r], lse_term[r], fm))
                )
                for r in range_constexpr(ACC16)
            ]
        )

    def dscores(self, p_list, v_dp, delta, keep=None):
        """`dS = P * (dP - delta)`. `dP` is unscaled; `sm_scale` belongs to dK.

        **Under dropout `dP` picks up `keep` and the survivor scale and `P`
        does not** -- the softmax denominator is the undropped sum, so the `P`
        outside the bracket is the undropped one. See the 32-row family's
        `dscores` for the derivation; the two differ only in the element count.
        """
        values = [Vec(v_dp)[r] for r in range_constexpr(ACC16)]
        if const_expr(keep is not None):
            s = self.ctx_ref.c_dropout_scale
            values = [
                keep[r].select(_fmul(values[r], s, self.fm_fast), self.c_zero_f)
                for r in range_constexpr(ACC16)
            ]
        return [
            _fmul(p_list[r], _fsub(values[r], delta[r], self.fm_fast), self.fm_fast)
            for r in range_constexpr(ACC16)
        ]

    def mask_if_clipped(self, p_list, tile_idx, sub):
        """Zero `P` outside the band, for the tiles that can be clipped.

        The 32-row family's `mask_if_clipped`, keyed on **this** family's row
        map rather than on that one's: element `i` of score sub-block `sub`
        holds `q = tile_base + (sub//2)*32 + (sub%2)*4 + 8*(lane//16) + i`,
        which is the same expression `load_row_values` addresses with. One
        table per family, no transcription -- and masks are
        exactly the code that tempts a second copy.

        The mask goes on `P`, so `dS = P * (dP - delta)` and `dV += P . dO`
        both inherit it from one select per element, and no `-inf` enters the
        arithmetic. The predicate is wave-uniform and a superset test.

        **No transpose read is inside this region** -- the q-contracted GEMMs
        run in their own loop after the softmax. CDNA4 section 11.4 requires
        EXEC all 1s across `ds_read_b64_tr_b16`; that holds by construction, as
        no transpose read sits inside a divergent region, and nothing in this
        tree scans the object for it.
        """
        traits = self.traits
        ctx = self.ctx_ref
        lo_i32 = ctx.causal_lo_i32
        left_i32 = ctx.window_left_i32
        sub_base = (sub // 2) * MFMA16_K + (sub % 2) * 4
        pack = bool(getattr(self, "GQA_PACK", False))
        if const_expr(pack):
            # `gqa_pack`: element `i` is packed row `x = tile_base + sub_base + 8*(lane//16) + i`, whose q is `x - g*Sq`; the head
            # does not enter the mask. A run of four rows may straddle two heads, so q is split per element.
            x0 = (
                tile_idx * fx.Index(traits.BLOCK_Q)
                + fx.Index(sub_base)
                + (self.lane // fx.Index(MFMA16_M)) * fx.Index(8)
            )
            q_el = [
                fx.Int32(pack_split(self, x0 + fx.Index(i))[1])
                for i in range_constexpr(ACC16)
            ]
            # Wave-uniform q bounds of the whole tile: `BLOCK_Q` consecutive q inside one head; across a head boundary or into the
            # pad, the head's whole extent.
            r0 = tile_idx * fx.Index(traits.BLOCK_Q)
            g_lo, q_first, _ = pack_split(self, r0)
            g_hi, _, _ = pack_split(self, r0 + fx.Index(traits.BLOCK_Q - 1))
            one_head = g_lo == g_hi
            tile_q0 = fx.Int32(one_head.select(q_first, fx.Index(0)))
            tile_q1 = fx.Int32(
                one_head.select(
                    q_first + fx.Index(traits.BLOCK_Q - 1), self.pack_sq - fx.Index(1)
                )
            )
            kv_lo = fx.Int32(ctx.kv_start) + fx.Int32(ctx.wave_kv_offset_uni)
            need = tile_q0 < kv_lo + fx.Int32(MFMA16_M - 1) + lo_i32
            if const_expr(traits.WINDOW):
                need = need | (tile_q1 > kv_lo + left_i32)
        else:
            rel0 = (
                fx.Int32(tile_idx * fx.Index(traits.BLOCK_Q))
                + fx.Int32(sub_base)
                + fx.Int32(self.lane // fx.Index(MFMA16_M)) * fx.Int32(8)
                - fx.Int32(ctx.kv_row)
            )
            kv_lo = fx.Int32(ctx.kv_start) + fx.Int32(ctx.wave_kv_offset_uni)
            tile_q0 = fx.Int32(tile_idx * fx.Index(traits.BLOCK_Q))
            need = tile_q0 < kv_lo + fx.Int32(MFMA16_M - 1) + lo_i32
            if const_expr(traits.WINDOW):
                need = need | (
                    tile_q0 + fx.Int32(traits.BLOCK_Q - 1) > kv_lo + left_i32
                )
        zero_f = self.c_zero_f
        window = traits.WINDOW

        def _apply(vals):
            out = list(vals)
            for i in range_constexpr(ACC16):
                rel = (
                    (q_el[i] - fx.Int32(ctx.kv_row))
                    if const_expr(pack)
                    else rel0 + fx.Int32(i)
                )
                keep = rel >= lo_i32
                if const_expr(window):
                    keep = keep & (rel <= left_i32)
                out[i] = keep.select(fx.Float32(vals[i]), zero_f)
            return out

        @flyc.jit
        def _mask_if_needed(p_vec):
            out = p_vec
            if need:
                out = Vec.from_elements(
                    [
                        as_mlir_value(v)
                        for v in _apply([Vec(p_vec)[i] for i in range_constexpr(ACC16)])
                    ],
                    fx.Float32,
                ).ir_value()
            return out

        packed = Vec.from_elements(
            [as_mlir_value(fx.Float32(v)) for v in p_list], fx.Float32
        ).ir_value()
        return [Vec(_mask_if_needed(packed))[i] for i in range_constexpr(ACC16)]

    def pack_group(self, lo, hi):
        """One 32-wide B operand from a q group's two score accumulators.

        Concatenation, not a shuffle. The row permutation in `row_base16` is
        what makes the two halves land at `k = 8*(lane//16) + 0..3` and
        `+ 4..7`; without it these eight values would be four of this lane's
        and four of another quarter wave's. `_bf16_trunc_pack_v8` is the 32-row
        family's, unchanged -- which is the other reason `16x16x32` is the
        cheaper port.

        The pack leaves through `mfma_operand_wait_state`, which supplies the
        `v_cvt_pk_bf16_f32`-into-`v_mfma` wait state. This is the family the
        hazard was found in -- `block_dmodel=128`, `mfma_rows=16`, bf16,
        dropout, matrix bias -- and that docstring has the measurement.
        """
        return mfma_operand_wait_state(
            dualwave._bf16_trunc_pack_v8(
                self.traits, list(lo) + list(hi), elem_dtype=self.elem_dtype
            )
        )


class M16StoreHelper(dualwave.DualwaveKernelContext):
    """dK and dV: one 8-byte store per lane per 16-wide chunk.

    A lane holds four **contiguous** d of one KV row, so the store is direct.
    The 32-row accumulator holds four rows `8*(i//4) + 4*(lane//32) + (i%4)`
    apart and has to be assembled into 128 bits through `permlane32_swap`.
    """

    def _pack_out(self, vals):
        """Four f32 accumulator values as four output elements, in either dtype.

        **The one place in this family that was dtype-specific**, and the way
        it was wrong under f16 is the reason it is a method now: it emitted
        `cvt_pk_bf16_f32` and then *bitcast* the result to `elem_dtype`, so an
        f16 build would have written bf16 bit patterns reinterpreted as f16 --
        finite, wrongly scaled by about 2^112, and with no diagnostic. A cast
        that reinterprets rather than converts cannot fail loudly.

        `_o_pack_2dw` and `_bf16_trunc_pack_v8` in the production helpers
        already branch this way; this is the same branch, on the same trait,
        for the accumulator shape this family has. bf16 keeps the packed
        two-at-a-time instruction, so its emitted code is unchanged.
        """
        if const_expr(self.traits.DTYPE_STR == "bf16"):
            return Vec.from_elements(
                [
                    dualwave.rocdl.cvt_pk_bf16_f32(vals[2 * j], vals[2 * j + 1])
                    for j in range_constexpr(ACC16 // 2)
                ],
                fx.Int32,
            ).bitcast(self.elem_dtype)
        # f16: convert each value (round-to-nearest-even) and build the vector
        # directly. There is no `cvt_pk_f16_f32` pair form to use here.
        return Vec.from_elements(
            [fx.Float32(vals[i]).to(self.elem_dtype) for i in range_constexpr(ACC16)],
            self.elem_dtype,
        )

    def store_accs(self, accs, which, hdim):
        rsrc = self.dk_rsrc if which == "dk" else self.dv_rsrc
        stride_seq = self.stride_dk_seq_v if which == "dk" else self.stride_dv_seq_v
        oob_off = self.dk_oob_off if which == "dk" else self.dv_oob_off
        base = self.kv_row * stride_seq
        lane_d = (self.lane // fx.Index(MFMA16_M)) * fx.Index(ACC16)
        cols = MaskedAxis(fx.Index(hdim)) if const_expr(self.PADDED_HEAD) else None
        for c in range_constexpr(len(accs)):
            col = fx.Index(c * MFMA16_M) + lane_d
            vals = [Vec(accs[c])[i] for i in range_constexpr(ACC16)]
            packed = self._pack_out(vals)
            off = base + col
            if const_expr(self.PADDED_HEAD):
                # Four columns, all-or-nothing. A run starting at or past the
                # real extent is pushed out of the descriptor and dropped; the
                # 8xD input contract makes a run that starts inside it end
                # inside the caller's own pad.
                off = fx.Index(cols.valid(col).select(fx.Index(off), oob_off))
            buffer_ops.buffer_store(
                packed.ir_value(), rsrc, as_mlir_value(fx.Int32(off))
            )


class M16TileBody(dualwave.DualwaveKernelContext):
    """One streamed q tile at 16 rows per wave: four GEMMs, two barriers.

    The same `run` signature, barriers and prefetch discipline as the 32-row
    body, so the kernel's tile loop does not know which family it is driving.
    """

    def __init__(
        self,
        ctx,
        *,
        stream,
        reader,
        gemm,
        softmax,
        hdim_qk,
        hdim_vo,
        keep_fn=None,
        bias_fn=None,
    ):
        super().__init__(ctx)
        self.stream = stream
        self.reader = reader
        self.gemm = gemm
        self.softmax = softmax
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        # `philox_keep` and `bias_f32`, injected rather than imported: both are
        # family-neutral and live in the module that owns the builder, which
        # already imports this one. Passing them keeps the dependency one-way.
        # Each takes this family's own `q_rows`, which is the whole difference.
        self.keep_fn = keep_fn
        self.bias_fn = bias_fn

    def load_lse_term(self, tile_base, sub):
        """The LSE term of the exponent for the four rows sub-block `sub` holds: `-log2(e) * LSE`, added; or `LSE`, subtracted, in a bias build (the exponent scales by `log2e` last). Overridden by the merged kernel (LSE -inf select)."""
        if const_expr(self.traits.BIAS_TYPE):
            return self.softmax.load_row_values(
                self.ctx_ref.lse_rsrc, tile_base, sub, self.ctx_ref.c_one_f
            )
        return self.softmax.load_row_values(
            self.ctx_ref.lse_rsrc, tile_base, sub, self.ctx_ref.c_neg_log2e
        )

    def load_delta(self, tile_idx, tile_base, sub, do_base, do_scope):
        """`Delta` for the four rows sub-block `sub` holds. Overridden by the merged kernel (rowsum(dO * O) in place)."""
        return self.softmax.load_row_values(
            self.ctx_ref.delta_rsrc, tile_base, sub, self.ctx_ref.c_one_f
        )

    def run(self, tile_idx, buf_id, prefetch_idx, v_k, v_v, dv, dk):
        traits = self.traits
        ctx = self.ctx_ref
        q_base = ctx.q_buf_base(buf_id)
        do_base = ctx.do_buf_base(buf_id)
        q_scope = dualwave._dualwave_lds_scope("k", buf_id)
        do_scope = dualwave._dualwave_lds_scope("v", buf_id)
        tile_base = tile_idx * fx.Index(traits.BLOCK_Q)
        n_grp = q_groups(traits)
        n_chunks = d_chunks16(traits)

        dualwave._waitcnt_vm_n(
            (traits.NUM_STREAM_BUFFERS - 1) * (ctx.NUM_DMA_K + ctx.NUM_DMA_V)
        )
        dualwave._sched_barrier(0)
        dualwave._s_barrier()

        # -- GEMMs 1 and 2 plus the softmax, one 32-row q group at a time.
        #    Always grouped this way, unlike the 32-row body's `TIGHT_REGISTERS`
        #    knob: a group's live set is two 16x16 accumulators, a quarter of
        #    what the 32-row body holds, so there is no trade to make here.
        p_packs = [None] * n_grp
        ds_packs = [None] * n_grp
        for g in range_constexpr(n_grp):
            p_half = []
            ds_half = []
            for h in range_constexpr(2):
                sub = 2 * g + h
                s = self.gemm.contract_d(
                    self._reader_for(q_base, q_scope, sub, self.hdim_qk), v_k
                )
                dp = self.gemm.contract_d(
                    self._reader_for(do_base, do_scope, sub, self.hdim_vo), v_v
                )
                lse_term = self.load_lse_term(tile_base, sub)
                # Folded into the exponent, so it lands before `P` exists.
                bias2 = None
                if const_expr(traits.BIAS_TYPE):
                    bias2 = self.bias_fn(ctx, self.softmax.q_rows(tile_idx, sub))
                p_list = self.softmax.probabilities(s, lse_term, bias2)
                if const_expr(traits.CAUSAL):
                    p_list = self.softmax.mask_if_clipped(p_list, tile_idx, sub)
                delta = self.load_delta(tile_idx, tile_base, sub, do_base, do_scope)
                # After the row sum, before the accumulation (the ordering the 32-row family uses).
                keep = None
                if const_expr(traits.ENABLE_DROPOUT):
                    keep = self.keep_fn(ctx, self.softmax.q_rows(tile_idx, sub))
                ds_half.append(self.softmax.dscores(p_list, dp, delta, keep))
                if const_expr(traits.ENABLE_DROPOUT):
                    p_list = [
                        keep[r].select(fx.Float32(p_list[r]), ctx.c_zero_f)
                        for r in range_constexpr(ACC16)
                    ]
                p_half.append(p_list)
            p_packs[g] = self.softmax.pack_group(p_half[0], p_half[1])
            ds_packs[g] = self.softmax.pack_group(ds_half[0], ds_half[1])

        # -- GEMMs 3 and 4. One transposed read pair per (chunk, group); each
        #    pack is consumed by one MFMA and dies.
        for c in range_constexpr(n_chunks):
            dv[c] = self.gemm.contract_q(
                self.reader.read_column_chunk(do_base, do_scope, c), p_packs, dv[c]
            )
        for c in range_constexpr(n_chunks):
            dk[c] = self.gemm.contract_q(
                self.reader.read_column_chunk(q_base, q_scope, c), ds_packs, dk[c]
            )

        dualwave._s_waitcnt(traits.LGKMCNT_0_ONLY)
        dualwave._sched_barrier(0)
        dualwave._s_barrier()
        self.stream.stage_q_tile(prefetch_idx, buf_id)
        self.stream.stage_do_tile(prefetch_idx, buf_id)
        return dv, dk

    def _reader_for(self, base, scope, sub, hdim):
        return lambda ks: self.reader.read_row_pack(base, scope, sub, ks, hdim)
