# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""dK / dV for gfx950 -- AOTriton's `bwd_kernel_dk_dv` on the dualwave helpers.

Dense and varlen, causal and windowed, MHA and GQA, with dropout and bias, in
bf16 and f16, over the forward's whole head_dim ladder with the 8xD input
contract and padded heads.

--- The whole design in one paragraph --------------------------------------

**dK/dV is the forward loop transposed.** K and V stay resident in registers
while Q and dO stream past through LDS, exactly mirroring the forward's
resident Q and streaming K/V. Every helper in `fmha_dualwave_gfx950.py` was
written against "a tensor, its bounds and where it lands" rather than against
K or V by name, so the roles swap by re-pointing four descriptors and nothing
else. The two tensor slots the staging machinery calls *K* and *V* carry **Q**
and **dO** here, and `BLOCK_M` / `BLOCK_N` / `VO_SHARDS` in the traits mean the
KV block, the Q tile and the dK/dV D-axis split -- see
`fmha_tuning_bwd_dkdv_gfx950`'s docstring for the full map.

--- Four GEMMs, and where each operand comes from ---------------------------

    S   = Q . K^T      contract d      A = Q  (LDS, row-major)  B = K  (regs)
    dP  = dO . V^T     contract d      A = dO (LDS, row-major)  B = V  (regs)
    dV^T = dO^T . P    contract q      A = dO (LDS, **column**) B = P  (regs)
    dK^T = Q^T  . dS   contract q      A = Q  (LDS, **column**) B = dS (regs)

`v_mfma_f32_32x32x16_bf16`'s A and B operands have the *same* per-lane layout
-- 32 outer rows selected by `lane % 32`, 16 contraction elements from
`lane // 32` and the element index -- so the output is `[A's row][B's row]`
with B's row landing on `lane % 32`. That symmetry is why the forward's K
reader and its Q loader produce interchangeable packs, and it is what makes
the first two GEMMs above a rename of `DualwaveGemmHelper.qk`.

**The two column-major operands are the forward's V read, unmodified.** The
forward stages V as `[token][d]` and reads it back through
`ds_read_b64_tr_b16` as `A[d][token]`, which is precisely what the
q-contracted GEMMs want from a `[q_row][d]` tile. So both streamed tiles are
staged in the *V* LDS shape and read two ways: `ds_read_b128` for the
row-major operand and the transpose path for the column-major one. Deriving a
second lane map was the alternative; reusing the validated one leaves a single
lane map to maintain.

--- What binds at the wide rungs, and what does not -------------------------

**LDS never forces `D_STAGES` here.** A staged slot is `68 * head_dim`
elements, so two tensors single-buffered are `272 * head_dim` bytes -- 139264
at head_dim 512, inside the 163840 cap. The second buffer is the only thing
LDS ever costs, and `_with_buffers` spends it while it can afford to.

**Registers do bind**, because a wave holds two accumulators of `d/2` VGPRs
each on top of `d/4` of resident K and `d/4` of V. Two levers answer it, in
the order they were found: `(num_waves, waves_per_eu)` as a pair, which is free and
decided head_dim 128; and then `DKV_SHARDS`, which divides the accumulator
pair across waves that write disjoint D columns -- no cross-wave reduction, no
extra barrier, at the price of every shard recomputing S and dP.

--- Padded heads: which operand is masked, and why it is the cheap one ------

The forward masks Q once in its prologue and K on the hot path, and pays for
the second one. Here the roles are swapped, so **the cheap masks are K and V**
-- resident, masked once, every step. Q and dO are the hot ones and are masked
only on the k-steps that can contain pad, which `HDIM_QK_FLOOR` bounds to two
on the 32-spaced ladder.

Only the two *d-contracted* GEMMs need masked operands at all. In `dV` and
`dK` the head dim is the **output** axis, so a pad column of dO can only reach
a pad column of dV, and the store suppresses it by address.

--- Why nothing else is masked ----------------------------------------------

Dense and non-causal, so the only edge is the ragged tail, and the buffer
descriptors already answer it. Q and dO are bounded at `seqlen_q` rows, so a
staged row past the end reads **zero**; K and V at `seqlen_kv`, and dK/dV
stores past it are dropped by the same bound. A padding q row therefore gets
`S = 0` and `LSE = 0` (also out of its buffer), hence `P = exp2(0) = 1` -- and
contributes `1 * dO = 0` to dV and `dS * Q = 0` to dK. Finite, exact, and no
`scf.if` anywhere near a transpose read, which is undefined under a divergent
EXEC.

--- Scope ------------------------------------------------------------------

Causal and windows, varlen, dropout, the bias *input* and GQA are all
implemented; there is no half-implemented arm anywhere below. `paged` is not a
knob of this kernel.

GQA is a group *fold*, not a second program: one workgroup owns a KV head and
walks the query heads of its group, and the dK/dV accumulators -- zeroed once,
before the loop -- carry the whole group sum in f32. What that costs is grid
width, `num_head_q` becoming `num_head_k`; see `retarget_q_head` for the
addressing and the tuning module for the occupancy measurement.

There is no `dB` here and never will be: `dB = dS`, which is materialised per
`(q, k)` element only in the dQ kernel. The bias is an input on this side.

**Bias and causal/window are mutually exclusive**, and the rule is semantic
rather than a gap: a bias *is* an attention mask, so pairing it with a
positional one asks which wins where they disagree and there is no answer.
`make_traits` raises on the pair, the forward raises on it in the same words,
AOTriton disables it and torch's math backend raises by name -- so a backward
that accepted the pair would also have no forward to produce its LSE.

--- Tensor argument order is the ABI ------------------------------------------

    q, k, v, b, do, dk, dv, lse, delta

Four groups, and the grouping is the mnemonic: the **forward's inputs**
(`q, k, v, b`), then the **backward's input** (`do`), then this kernel's
**outputs**, then the **lower-rank** tensors (`lse`, `delta`, both rank 2).

It is also AOTriton's order -- `bwd_kernel_dq(Q, K, V, B, sm_scale, DO, DQ, DB,
L, D, ...)` and `bwd_kernel_dk_dv(Q, K, V, B, sm_scale, DO, DK, DV, L, D, ...)`
-- so a reader moving between the Triton reference and this file does not have
to re-derive the mapping, and neither does anyone eventually dispatching the
compiled hsaco directly.

`b` sits in the forward-input group although the first builds never read it, so
adding bias did not move the wire format. After `delta`, at the end, is where an
unused argument naturally lands, which is why the grouping is written down
rather than left to accrete.
"""

import weakref
from dataclasses import dataclass, replace as _dc_replace
from types import SimpleNamespace as _WireArgs

from kernels.attention_bwd_cdna import fmha_abi_gfx1201 as abi
from kernels.attention_bwd_cdna import fmha_bwd_dkdv_m16_gfx950 as m16
from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha
from kernels.attention_bwd_cdna.fmha_common_gfx1201 import MaskedAxis
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import _fadd, _fmul, _fsub
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvGmemToLdsLoader,
    ParitySoftmaxHelper,
    traits_cache_key,
    _bias_slab_num_records_bytes,
    _ds_read_tr_v4f16_imm,
    _score_column_runs,
    _slab_span_elems,
    _v_imm_lo,
    exp2_wait_state,
    mfma_operand_wait_state,
    wire_ptr,
    wire_view,
)
from kernels.attention_bwd_cdna.fmha_tuning_bwd_dkdv_gfx950 import (
    BwdDkDvInputMetadata,
    bwd_dkdv_knobs,
    check_launch_limits,
)
from kernels.attention_bwd_cdna.gfx950_standalone import buffer_ops, dualwave
from kernels.attention_bwd_cdna import philox
from kernels.attention_bwd_cdna.philox import Philox, keep_mask

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

__all__ = [
    "KERNEL_NAME",
    "build_fmha_bwd_dkdv_gfx950_module",
    "build_fmha_bwd_dkdv_gfx950_module_primary",
]

KERNEL_NAME = "fmha_bwd_dkdv_gfx950_kernel"

_s_barrier = dualwave._s_barrier
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier
_waitcnt_vm_n = dualwave._waitcnt_vm_n

_COMPILED = weakref.WeakKeyDictionary()


def _pin_sgpr(value):
    """An opaque scalar copy of `value`, so it is computed here, in the prologue, and not sunk to its use.

    `gqa_pack` carries the dK/dV store scalars (two descriptors, two row strides, two out-of-range offsets) to the
    epilogue across the whole tile walk. Left as the `i64` kernarg strides they derive from, those strides keep a
    `s_load_dwordx16` tuple (kernarg bytes 248..311) live across the loop, and the backend's lane-spill of that tuple
    can lose a staging copy (ROCm/FlyDSL#1087): the epilogue then multiplies a row by a stale register. A side-effecting
    inline asm cannot be sunk, so the tuple dies in the prologue.
    """
    ir_val = as_mlir_value(value)
    return dualwave.llvm.inline_asm(
        ir_val.type, [ir_val], "", "=s,0", has_side_effects=True
    )


_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}

# `_causal_pair_thresholds(False)` flattened is `8 * (r // 4) + (r % 4)`, which
# is the MFMA accumulator's row map, and grouping it into runs gives four spans
# of four. So one `buffer_load` of 4 f32 covers one span of LSE (or delta) rows
# exactly, and the row set a lane needs is 4 loads per accumulator half. The
# forward derives the same runs for its bias reads; deriving them once keeps the
# two consumers in step.
_ROW_RUNS = _score_column_runs(False)

# The same runs flattened: `_ROW_THRESHOLDS[r]` is the q row accumulator element
# `r` holds, before the tile base and the lane term. Derived from `_ROW_RUNS`
# rather than written out, because a mask keyed on a lane->(row, col) map is
# exactly the code that tempts a second transcription -- and the 32-row and
# 16-row families each get this from their own map, never from each other's.
_ROW_THRESHOLDS = [0] * 16
for _e0, _c0, _w in _ROW_RUNS:
    for _j in range(_w):
        _ROW_THRESHOLDS[_e0 + _j] = _c0 + _j


def _stream_row_read_base(traits, lane_mod_32, lane_div_32):
    """Per-lane LDS base for the row-major read of a staged tile.

    `fmha_dualwave_gfx950._k_read_base` with `STREAM_LINE_STRIDE` in place of
    `SMEM_K_LINE_STRIDE`: both streamed tiles are V-shaped (see
    `BwdDkDvTraits.STREAM_LINE_STRIDE`), and the line stride is the only term
    of the K read that knows which shape it is addressing.
    """
    return (
        (lane_mod_32 % traits.SMEM_N_RPT) * traits.STREAM_LINE_STRIDE
        + (lane_mod_32 // traits.SMEM_N_RPT) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _stream_ks_offset(traits, ks):
    """LDS offset of MFMA k-step `ks` within a staged tile, in elements."""
    per_band = traits.K_STEPS_PER_BAND
    return (ks // per_band) * (traits.SMEM_N_RPT * traits.STREAM_LINE_STRIDE) + (
        ks % per_band
    ) * traits.K_STEP_QK


def _masked_ks_steps(traits, hdim_floor):
    """The k-steps whose D columns can reach past the real head dim.

    A build serves `(HDIM_QK_FLOOR, BLOCK_DMODEL]` and the dispatcher enforces
    it, so a step whose columns all lie at or below the floor is reading real
    data whatever the runtime extent. On the 32-spaced ladder consecutive rungs
    are 32 apart and `K_STEP_QK` is 16, so exactly two steps survive at every
    width -- 2 of 32 at head_dim 512 rather than 32.

    The forward found this to be the whole cost of a padded head: masking every
    step ran 27-54% below the rung's native rate, near-independently of how
    much pad there was, which is what identified the masking rather than the
    wasted MFMA columns.
    """
    return [
        ks
        for ks in range(traits.K_STEPS_QK)
        if (ks + 1) * traits.K_STEP_QK > hdim_floor
    ]


# DPP `quad_perm` controls: lane `i` of every group of four reads lane `i ^ 1`
# (`[1,0,3,2]`) or `i ^ 2` (`[2,3,0,1]`).
_DPP_QUAD_XOR1 = 0xB1
_DPP_QUAD_XOR2 = 0x4E


def _quad_xor(x, ctrl):
    """`x` of the lane at `lane ^ k` within each quad, one `v_mov_b32_dpp` (`ctrl` names `k`)."""
    v = fx.Int32(x)
    return fx.Int32(
        rocdl.update_dpp(T.i32, v.ir_value(), v.ir_value(), ctrl, 0xF, 0xF, False)
    )


def _pred_select(pred, a, b):
    return fx.Int32(pred.select(a, b))


def quad_transpose(words, upper, odd, xor, select):
    """Transpose the 4x4 matrix whose row `s` is `words` on lane `s` of each quad.

    Afterwards lane `t` holds, in position `e`, what lane `e` held in position
    `t`. It is two block-swap stages (lane bit 1 against word bit 1, then lane
    bit 0 against word bit 0); a stage sends one word pair across a quad XOR
    and keeps the other, which is a select before the move and two after.

    `upper` / `odd` are the lane's bits 1 / 0 within its quad. `xor(x, ctrl)`
    reads `x` from lane `lane ^ k` of the quad (`ctrl` is `_DPP_QUAD_XOR2` for
    `k = 2` then `_DPP_QUAD_XOR1` for `k = 1`) and `select(pred, a, b)` is `a`
    where `pred` holds, `b` elsewhere; both are injected so a host-level oracle
    can run this same function.
    """

    def swap(lo, hi, pred, ctrl):
        moved = xor(select(pred, lo, hi), ctrl)
        return select(pred, moved, lo), select(pred, hi, moved)

    w0, w1, w2, w3 = words
    w0, w2 = swap(w0, w2, upper, _DPP_QUAD_XOR2)
    w1, w3 = swap(w1, w3, upper, _DPP_QUAD_XOR2)
    w0, w1 = swap(w0, w1, odd, _DPP_QUAD_XOR1)
    w2, w3 = swap(w2, w3, odd, _DPP_QUAD_XOR1)
    return [w0, w1, w2, w3]


def philox_keep(ctx, q_rows):
    """`keep` per accumulator element, given each element's absolute q row.

    **One philox implementation, two row maps.** The families differ only in
    which q row an element holds, so they pass that in and share everything
    else -- which is the contract's rule for anything keyed on a lane->(row,
    col) map, applied to the one piece of code that would otherwise be
    transcribed twice.

    A dK/dV lane owns **one KV column and many q rows**, the transpose of the
    forward, so the elements are `row_stride` apart in the stream and there is
    no run to draw as a span. A `philox_4x` call yields the four columns
    `4c .. 4c+3` of one row, and those are four adjacent lanes -- a quad --
    that all hold the same q rows. So the quad splits the work instead of
    repeating it: **lane `s` of the quad draws the call for the `s`-th row of
    each run of four** (`q_rows` is runs of four consecutive rows in both
    families), and `quad_transpose` hands every lane the word of that call
    that is its own column. One call per four elements, as in the forward,
    for a transpose of 12 selects and 4 DPP moves per four elements. Offsets
    and slots are unchanged, so the mask is the forward's bit for bit.

    The exchange reads the neighbouring lanes of the quad, so this runs with
    EXEC all 1s -- the same requirement the transpose reads carry. A lane's
    quad position is its KV column mod 4 (`philox_slot`): the wave's KV rows
    start at a multiple of four and lane `l` holds row `l % rows`.
    """
    rng = ctx.philox_rng
    if rng.randoms_per_offset != 4 or len(q_rows) % 4:
        raise ValueError(
            "philox_keep exchanges runs of four rows over a four-wide philox call"
        )
    kv = fx.Int64(ctx.kv_row)
    out = []
    for g in range_constexpr(len(q_rows) // 4):
        row = q_rows[4 * g] + ctx.philox_slot
        words = rng.u32(
            ctx.philox_seed,
            rng.grid_offset(
                ctx.philox_plane_base, ctx.philox_row_stride, fx.Int64(row), kv
            ),
        )
        out.extend(
            keep_mask(
                quad_transpose(
                    words,
                    ctx.philox_quad_upper,
                    ctx.philox_quad_odd,
                    _quad_xor,
                    _pred_select,
                ),
                ctx.idropout_p,
            )
        )
    return out


def bias_log2e(ctx, q_rows):
    """`bias * log2(e)` per accumulator element, given each element's absolute q row.

    **One bias read, two row maps**, the same arrangement `philox_keep` uses and
    for the same reason: the families differ only in which q row an element
    holds, so they pass that in and share the rest.

    The `* log2(e)` is here rather than at the use site because the exponent
    lives in the base-2 domain -- `P = exp2(qk_scale*S - log2e*LSE)` -- and a
    bias in natural units has to cross into it. This is AOTriton's
    `qk += bias * 1.44269504089`.

    **`fm_bias`, not `fm_fast`, and it is load-bearing.** `fm_fast` is MLIR's
    `fast`, which carries `ninf` -- a licence to assume no operand is infinite.
    A bias entry of `-inf` is precisely how a caller spells "never attend
    here", so this is the one place in this kernel where a real infinity enters
    *arithmetic* rather than a select: the causal mask writes zeros through
    `select` and the KV tail through the descriptor bound, neither of which
    fastmath touches. The forward records `ninf` silently deleting a KV tail
    mask on gfx1201; the same licence taken up by a later pass would delete a
    caller's `-inf` here. The flag rides the whole exponent chain in a bias
    build (see `probabilities`), which costs a non-bias build nothing because
    `BIAS_TYPE` is a build axis.

    **The lane gets no vector and the wave gets perfect coalescing**, which is
    the opposite of the dropout case and worth stating because the two look
    alike. A lane's 16 elements are 16 q rows, `stride_b_seq_q` apart, so there
    is no run to draw as a span and it is 16 scalar loads. But the 32 lanes of
    a row group hold 32 *consecutive* KV columns of the same q row, and the KV
    axis is the bias's contiguous one -- so each of those 16 loads is one fully
    coalesced 64-byte line across the group, not 32 scattered dwords. The
    forward gets its vector per lane instead; both end up reading each bias
    element exactly once.

    **Loads first.** Every load of the call is issued before the first convert/multiply, so the loads overlap and drain with staggered
    `vmcnt(N)` waits; a load followed by its own convert serialises behind a full `s_waitcnt vmcnt(0)` per element.
    """
    rsrc = ctx.bias_rsrc
    pitch = fx.Index(ctx.stride_b_seq_q)
    col = fx.Index(ctx.kv_row)
    log2e = fx.Float32(dualwave._LOG2E)
    fm_bias = fx.arith.FastMathFlags.contract | fx.arith.FastMathFlags.reassoc
    # `vec_width=1` returns a **scalar** of the element type, not a
    # one-lane vector -- so it is wrapped directly rather than through
    # `Vec(...)[0]`, which is what the forward's width-4 and width-8 reads
    # use and which raises here.
    ones = [
        buffer_ops.buffer_load(
            rsrc,
            as_mlir_value(fx.Int32(q * pitch + col)),
            vec_width=1,
            dtype=ctx.elem_dtype,
        )
        for q in q_rows
    ]
    return [
        _fmul(fx.Float32(ctx.elem_dtype(one).to(fx.Float32)), log2e, fm_bias)
        for one in ones
    ]


def bias_log2e_packed(ctx, q_rows):
    """`bias_log2e` for `gqa_pack`: `q_rows` are flat rows of the packed axis, each addressed `g * stride_b_head + q * pitch`.

    A row past the packed extent is redirected to `pack_bias_oob` (at or past the descriptor's `num_records`), so it reads zero as a
    row past `seqlen_q` does in the unpacked path. Every load is issued before the first convert, as in `bias_log2e`.
    """
    rsrc = ctx.bias_rsrc
    pitch = fx.Index(ctx.stride_b_seq_q)
    head = fx.Index(ctx.stride_b_head)
    col = fx.Index(ctx.kv_row)
    log2e = fx.Float32(dualwave._LOG2E)
    fm_bias = fx.arith.FastMathFlags.contract | fx.arith.FastMathFlags.reassoc

    def offset(x):
        g, q, valid = m16.pack_split(ctx, x)
        return fx.Index(valid.select(g * head + q * pitch, ctx.pack_bias_oob)) + col

    ones = [
        buffer_ops.buffer_load(
            rsrc, as_mlir_value(fx.Int32(offset(x))), vec_width=1, dtype=ctx.elem_dtype
        )
        for x in q_rows
    ]
    return [
        _fmul(fx.Float32(ctx.elem_dtype(one).to(fx.Float32)), log2e, fm_bias)
        for one in ones
    ]


def _lds_pack_read(traits, lds_ptr, elem_idx, scope_name, pack_type):
    """One 128-bit LDS read of 8 bf16, alias-scoped to the buffer it touches.

    `dualwave._load_k_pack_aligned` with the scope name as a parameter. It
    derives its own as `lds_k{buf}`, which is right for the forward's one
    row-major reader and wrong here, where two tensors are staged and the
    second lives in the `v` scopes. The alias scopes are not decoration: the
    backend drains *all* outstanding `buffer_load ... lds` before any DS read
    that may alias one, and that drain was 26% of the forward's runtime.
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


class BwdDkDvKernelContext(ParityKernelContext):
    """Parity context whose resident tensor is K/V and whose stream is Q/dO.

    Inherits the BHSD stride ABI, the runtime head counts, the varlen decode
    and the philox prologue by subclassing rather than porting.

    Six things move, and they are all descriptors, bounds or indices:

    - `strides` carries **18** slots: Q, K, V, dO, dK, dV. Only the first nine
      go to the base class, which is all it addresses; dO, dK and dV are held
      here under their own names, with `stride_do_*`, `stride_dk_*` and
      `stride_dv_*`. **There is no `O` here.** Handing dK to the base's `O`
      slot would make `stride_o_*` the strides of whichever tensor happened
      to be in the slot; this kernel writes dK and dV, both KV-row-shaped,
      and builds `dk_div` and `dv_div` for them directly.
    - The staging machinery's K slot carries Q and its V slot carries dO.
    - Both staged tiles are V-shaped, so the LDS bases and the row-major read
      base are recomputed against `STREAM_LINE_STRIDE`.
    - The tile loop walks **q** tiles, so `init_tile_bounds` counts them from
      `seqlen_q`.
    - `wave_id` splits into a KV row block and a D-axis shard.
    - LSE and delta get f32 row resources, which the forward has no reader for.
    """

    # Perf knob, set by the builder right after construction (before any `init_*`); see `BwdDkDvKnobs`.
    GQA_PACK = False

    def __init__(self, traits, *, strides, DO, DK, DV, Delta, **kwargs):
        # Q, K, V only. The base's `O` slot stays empty -- this kernel has no
        # O -- so it builds no O view and `stride_o_*` stays zero and unread.
        super().__init__(traits, strides=strides[:9], **kwargs)
        self.DO = DO
        self.DK = DK
        self.DV = DV
        self.Delta = Delta
        (
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.stride_dk_batch,
            self.stride_dk_head,
            self.stride_dk_seq,
            self.stride_dv_batch,
            self.stride_dv_head,
            self.stride_dv_seq,
        ) = strides[9:18]

    # -- LDS ---------------------------------------------------------------

    def init_lds(self, shared_storage):
        """Allocate the staged tiles, keeping the base class's attribute names.

        `lds_kv_base_idx` / `lds_kv_base_ptr` are read by every DMA and LDS
        helper in the stack, so they stay -- only the storage field is renamed
        to say what it holds.
        """
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.stream.ptr))
        self.lds_kv_base_ptr = lds.stream.ptr.llvm_ptr
        self.lds_bt_base_ptr = None

    def q_buf_base(self, buf_id):
        """First element of the Q tile in stream buffer `buf_id`."""
        return (2 * buf_id) * self.traits.STREAM_TILE_ELEMS

    def do_buf_base(self, buf_id):
        """First element of the dO tile in stream buffer `buf_id`."""
        return (2 * buf_id + 1) * self.traits.STREAM_TILE_ELEMS

    def k_dma_base(self, buf_id, d):
        """m0 for the Q staging DMA. The forward's K slot; see the class docstring."""
        return self._dma_m0(self.q_buf_base(buf_id), self.traits.STREAM_LINE_STRIDE, d)

    def v_dma_base(self, buf_id, d):
        """m0 for the dO staging DMA. The forward's V slot."""
        return self._dma_m0(self.do_buf_base(buf_id), self.traits.STREAM_LINE_STRIDE, d)

    def dma_m0_table(self, base_fn, count):
        """One row per *stream buffer*, not the base class's fixed two.

        At `NUM_STREAM_BUFFERS == 1` the second row would address LDS past the
        allocation. It is never issued, so it folds away -- but a table that
        contains an illegal address is a thing a later reader has to reason
        about, and building the right number is free.
        """
        return tuple(
            tuple(base_fn(buf, d) for d in range(count))
            for buf in range(self.traits.NUM_STREAM_BUFFERS)
        )

    def init_lds_read_bases(self):
        super().init_lds_read_bases()
        traits = self.traits
        self.stream_row_read_base = _stream_row_read_base(
            traits, self.lane_mod_32, self.lane_div_32
        )
        # The column-major base is the forward's V one *unchanged*: the tile is
        # V-shaped and the read is the same instruction sequence, so the
        # validated lane map carries over rather than being re-derived.
        col_base = self.v_lds_read_base_per_lane
        if const_expr(traits.DKV_SHARDS > 1):
            # The shard's D offset folds into the base rather than being added
            # per read, because `_v_imm_lo` needs a *compile-time* chunk index
            # and `shard_id` is `wave_id % SHARDS`. `_v_dc_offset` splits as
            # `_v_dc_offset(a + i) = (a // PER_BAND) * PAIR + _v_dc_offset(i)`
            # whenever `a` is a multiple of `D_CHUNKS_PER_BAND`, which
            # `make_traits` enforces for a shard's first chunk.
            per_band = traits.D_CHUNKS_PER_BAND
            col_base = col_base + self.dkv_shard_id * (
                (traits.D_CHUNKS_PER_SHARD // per_band)
                * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE
            )
        self.stream_col_read_base = col_base

    # -- indices -----------------------------------------------------------

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_do_seq_v = fx.Index(self.stride_do_seq)
        self.stride_dk_seq_v = fx.Index(self.stride_dk_seq)
        self.stride_dv_seq_v = fx.Index(self.stride_dv_seq)

    def init_thread_mapping(self):
        super().init_thread_mapping()
        traits = self.traits
        # Aliases, not new values: `grid.y` selects a KV block here and the
        # base class's `q_*` names already hold exactly that arithmetic.
        self.kv_block_idx = self.q_block_idx
        self.kv_start = self.q_start
        # **`grid.x` enumerates the KV head, not the query head** -- the one
        # grid change GQA makes. The base class decomposes `h_idx` against
        # `num_head_k` and recomposes against the group size, so with an extent
        # of `num_head_k` its `group_id` is always 0 and its `q_head_idx` lands
        # on the *first* query head of the group. That is the base the loop
        # walks from, and the base class's own mapping is what supplies it --
        # no second transcription of the head remap.
        #
        # The KV block stays on `grid.y`. gfx1201's fold moves it to the fast
        # axis for causal load balance; gfx950 measured that exact change at
        # 12-15% slower at every rung and deleted the knob (eight XCDs with
        # separate L2s), so the fold does not get to bring it back in.
        self.gqa_q_head_base = self.q_head_idx
        if const_expr(traits.DKV_SHARDS > 1):
            # With `SHARDS` waves per KV row block, `wave_id` no longer names a
            # row block -- `wave_id // SHARDS` does, and the remainder picks
            # which D columns of dK/dV the wave owns. The base class set
            # `wave_q_offset` from the raw wave id; this is the same correction
            # the forward's sharded context makes for `VO_SHARDS`.
            self.dkv_shard_id = self.wave_id % traits.DKV_SHARDS
            self.wave_q_offset = (
                self.wave_id // traits.DKV_SHARDS
            ) * traits.ROWS_PER_WAVE
        else:
            self.dkv_shard_id = fx.Index(0)
        self.wave_kv_offset = self.wave_q_offset
        # The same offset from the *uniform* wave id. The causal predicate is
        # wave-uniform and must stay so: built from `wave_id` it would be a
        # lane-varying compare, and the branch it feeds would then narrow EXEC
        # -- which is the one thing the transpose reads cannot tolerate.
        rows_uni = traits.ROWS_PER_WAVE
        if const_expr(traits.DKV_SHARDS > 1):
            self.wave_kv_offset_uni = (self.wave_id_uni // traits.DKV_SHARDS) * rows_uni
        else:
            self.wave_kv_offset_uni = self.wave_id_uni * rows_uni
        # Global first D column of this wave's accumulators. Runtime, because
        # the shard comes from `wave_id`; zero and folded away when unsharded.
        self.dkv_col_base = self.dkv_shard_id * fx.Index(
            traits.D_CHUNKS_PER_SHARD * traits.D_CHUNK
        )

    def init_philox(self):
        """Seed, counter and this workgroup's plane. Prologue-only.

        `ParityKernelContext.init_philox` with the **report removed**, and that
        is the whole difference. The forward *writes* the `(seed, offset)` it
        drew from, because under graph capture the effective offset is
        `*offset1 + offset2` -- a sum formed on the device from a counter the
        host cannot read without synchronising. The backward is the consumer of
        that record; writing it back would be a second opinion about a fact the
        forward already settled, so this kernel has no output slots for it.

        Everything else is the forward's, called through the same
        `Philox.grid_plane`: the plane is `(batch, head)`, and its arguments
        are `max_seqlen_q` / `max_seqlen_k` rather than the tile geometry or
        the per-sequence lengths. **That is the reproducibility contract** --
        the mask is a function of element coordinates alone, so neither a
        re-tuned tile size nor a varlen batch can move a single random.
        """
        if const_expr(not self.traits.ENABLE_DROPOUT):
            return
        self.philox_rng = Philox.for_arch("gfx950")
        self.philox_seed = fmha.philox_seed_value(self.philox_seed_ptr)
        self.philox_offset_base_v = fmha.philox_offset_base(
            self.philox_offset1, self.philox_offset2
        )
        # A lane owns one KV column for the whole kernel, so its position in
        # the quad of lanes that share a philox call (the column mod
        # `randoms_per_offset`) is loop-invariant -- and it stays so across a
        # GQA group, since the group moves the *query* head and the slot is a
        # function of the KV column alone. `philox_keep` reads it as the row
        # offset it draws and as the two predicates of its quad transpose.
        self.philox_slot = fx.Index(self.kv_row) % fx.Index(
            self.philox_rng.randoms_per_offset
        )
        self.philox_quad_upper = self.philox_slot >= fx.Index(2)
        self.philox_quad_odd = (self.philox_slot & fx.Index(1)) == fx.Index(1)
        # The plane is `(batch, q head)` and therefore moves with the group, so
        # it is bound alongside the other query-side state. This call gives the
        # prologue a plane for the group's first head; the loop rebinds it.
        #
        # `_bind_philox_plane` and not the whole of `bind_q_head`: the plane is
        # the only thing missing at this point, and rebuilding six descriptors
        # to reach it emitted a second copy of all of their address arithmetic
        # for the canonicaliser to fold back out.
        self._bind_philox_plane()

    def compute_active_guard(self):
        """Whether this workgroup's KV block exists in *this* sequence.

        The grid is sized from `max_seqlen_k`, so a short sequence dispatches
        blocks past its own keys. The base class's guard tests the *Q* extent,
        which is the right question for the forward and the wrong one here.

        Unconditional: a dense call has `seqlen_kv_v == max_seqlen_k`, so the
        compare is true for every block the grid dispatches and costs one
        scalar instruction to discover it.

        Not needed for correctness -- the KV descriptors bound at
        `seqlen_kv` rows, so a dead block reads zeros and its stores are
        dropped -- but it is a lot of wasted work at a ragged batch. The
        predicate is workgroup-uniform, so the branch is scalar and EXEC stays
        all 1s across the transpose reads inside it.
        """
        return self.kv_start < self.seqlen_kv_v

    def init_kv_row(self):
        """The KV rows this wave owns: `ROWS_PER_WAVE` of them, at `lane % rows`.

        `init_q_row`'s arithmetic at 32 rows, where `lane_mod_32` is already the
        row selector. The 16-row family's accumulator puts `n` on `lane % 16`,
        so the row it names is a different one.
        """
        self.init_q_row()
        rows = self.traits.MFMA_ROWS
        if const_expr(rows == 32):
            self.kv_row_in_block = self.q_row_in_block
        else:
            self.kv_row_in_block = self.wave_kv_offset + self.lane % fx.Index(rows)
        self.kv_row = self.kv_start + self.kv_row_in_block

    # -- descriptors -------------------------------------------------------

    def init_descriptors(self, **kwargs):
        """Six tensor views plus two f32 row resources.

        `super()` runs first for the state that is not about addressing --
        `delta_i32`, `buf_flags_i32`, `elem_ir` -- and its four dense views are
        replaced below. They are pure descriptor arithmetic with no side
        effects, so they fold away; the forward's `init_descriptors` makes the
        same trade for the same reason.
        """
        super().init_descriptors(**kwargs)
        traits = self.traits
        # **Each side owns its own row origin and its own batch index.** Under
        # `0x040B` -- packed Q against a *batched* KV cache -- Q is stacked
        # (batch 0, large row offset) and K is batched (batch z, no row offset)
        # in the same call. Four of the five modes agree, so reusing Q's index
        # for K reads batch 0 of the cache for every sequence and only that one
        # mode exposes it.
        self.q_row_off = self.varlen_q_row_off
        self.kv_row_off = self.varlen_kv_row_off
        # The staged tiles address from token 0 of the slab; the tile offset
        # rides in the DMA's `soffset`, which is what `_kv_tile_addr` produces.
        self.q_gmem_elem_offset = fx.Index(0)
        self.kv_gmem_elem_offset = fx.Index(0)

        # Everything keyed on the *query* head, built once here for the first
        # head of the group and rebuilt per group iteration under GQA. The
        # invariants come first and are hoisted deliberately -- see
        # `_init_q_head_invariants`.
        self._init_q_head_invariants()
        self.bind_q_head(fx.Index(0))

        # Resident, and the two outputs: all four are (batch, kv head) slabs
        # bounded at the last real element of their `seqlen_kv`th row. K and dK
        # are `hdim_qk` wide, V and dV `hdim_vo` -- the extents are what bounds
        # the slab, not the row stride; see `_slab_span_elems`.
        self.k_res_div = self._slab_view(
            self.K,
            self.stride_k_batch,
            self.stride_k_head,
            self.stride_k_seq,
            self.kv_row_off,
            self.kv_head_idx,
            self.seqlen_kv_v,
            self.hdim_qk,
            batch_idx=self.kv_batch_idx,
        )
        self.v_res_div = self._slab_view(
            self.V,
            self.stride_v_batch,
            self.stride_v_head,
            self.stride_v_seq,
            self.kv_row_off,
            self.kv_head_idx,
            self.seqlen_kv_v,
            self.hdim_vo,
            batch_idx=self.kv_batch_idx,
        )
        self.dk_div = self._slab_view(
            self.DK,
            self.stride_dk_batch,
            self.stride_dk_head,
            self.stride_dk_seq,
            self.kv_row_off,
            self.kv_head_idx,
            self.seqlen_kv_v,
            self.hdim_qk,
            batch_idx=self.kv_batch_idx,
        )
        self.dv_div = self._slab_view(
            self.DV,
            self.stride_dv_batch,
            self.stride_dv_head,
            self.stride_dv_seq,
            self.kv_row_off,
            self.kv_head_idx,
            self.seqlen_kv_v,
            self.hdim_vo,
            batch_idx=self.kv_batch_idx,
        )

        # Raw bounded resources over the same four slabs, for the 16-row
        # family's 64-bit loads and stores. A `_slab_view` carries a copy atom
        # whose width is fixed at 128 bits; these take an element offset and a
        # width per access, and they bound identically, so a row past the
        # sequence reads zero and a store to one is dropped.
        self.k_res_rsrc = self._slab_rsrc(
            self.K,
            self.stride_k_batch,
            self.stride_k_head,
            self.stride_k_seq,
            self.hdim_qk,
        )
        self.v_res_rsrc = self._slab_rsrc(
            self.V,
            self.stride_v_batch,
            self.stride_v_head,
            self.stride_v_seq,
            self.hdim_vo,
        )
        _pin = bool(self.GQA_PACK)
        self.dk_rsrc = self._slab_rsrc(
            self.DK,
            self.stride_dk_batch,
            self.stride_dk_head,
            self.stride_dk_seq,
            self.hdim_qk,
            pin=_pin,
        )
        self.dv_rsrc = self._slab_rsrc(
            self.DV,
            self.stride_dv_batch,
            self.stride_dv_head,
            self.stride_dv_seq,
            self.hdim_vo,
            pin=_pin,
        )

        self.k_res_elem_base = self.kv_start * self.stride_k_seq_v
        self.v_res_elem_base = self.kv_start * self.stride_v_seq_v
        # At or past the end of each output's descriptor, so a store redirected
        # here is dropped by the hardware bound -- which is how the padded-head
        # D tail is suppressed without a branch. `_slab_span_elems` can only
        # shrink those descriptors below this, never grow one past it, so the
        # order the suppression needs holds for both the `_slab_view` and the
        # `_slab_rsrc` over the same slab.
        self.dk_oob_off = self.seqlen_kv_v * self.stride_dk_seq_v
        self.dv_oob_off = self.seqlen_kv_v * self.stride_dv_seq_v
        if const_expr(_pin):
            self.stride_dk_seq_v = fx.Index(
                fx.Int64(_pin_sgpr(fx.Int64(self.stride_dk_seq_v)))
            )
            self.stride_dv_seq_v = fx.Index(
                fx.Int64(_pin_sgpr(fx.Int64(self.stride_dv_seq_v)))
            )
            self.dk_oob_off = fx.Index(fx.Int64(_pin_sgpr(fx.Int64(self.dk_oob_off))))
            self.dv_oob_off = fx.Index(fx.Int64(_pin_sgpr(fx.Int64(self.dv_oob_off))))

    def _init_q_head_invariants(self):
        """Everything `bind_q_head` needs that does **not** move with the head.

        **This split is a scalar-pressure change and nothing more. Read the last
        paragraph before you attribute any correctness property to it.**

        The hoist is measured against the unhoisted form below. Whether it still
        pays on top of attacking the same liveness from the addressing side is
        not measured; retire it if an A/B on an idle gfx950 shows it inert.

        `bind_q_head` runs inside the GQA loop, whose trip count is a runtime
        value and so cannot fold away, which means every value it reads is live
        across the largest region in the kernel. Reading the nine Q/dO/bias
        strides there kept nine `i64` kernargs live across the whole tile walk.

        What makes the hoist possible is that all of it is affine in the head:
        `_slab_byte_base` moves by `stride_head` per head, `lse_row_addressing`'s
        base by `lse_row_step_per_head`, and the spans and bounds -- which read
        the *seq* strides, the wide ones -- do not move at all. So the loop needs
        one add per tensor and none of the strides.

        Measured over all 216 built configurations: `sgpr_spill_count` median
        97 -> 87, 180 improved / 32 worse / 4 unchanged, -2147 in total, and
        zero-spill builds 4 -> 16. VGPR is flat. That is the whole of the claim.

        **It does not fix the gfx950 lost-`s_mov_b64` miscompile, and it was
        wrong to expect it to.** The hypothesis was that the long stride live
        ranges were the precondition for the backend dropping a staging copy.
        They are not. Across the same 216 builds the defect went 42 -> 28, but
        the *set* churned: 32 cleared and **18 previously-clean builds newly
        acquired it**, 14 of those while their spill count went down. It also
        survives forcing scratch spilling instead of lane spilling
        (`amdgpu-spill-sgpr-to-vgpr=false`, which flags an identical 28) and
        both existing `_COMPILE_HINTS` flags, and 19 builds are flagged under
        every setting tried. So spill pressure is not the control variable and
        this file cannot reach the bug. ROCm/FlyDSL#1087 tracks it, with the
        ISA and a wheel-only reproducer; only a backend fix closes it.
        """
        elem_bytes = fx.Index(self.traits.BF16_BYTES)
        head0 = self.gqa_q_head_base

        # Streamed. Bounded at `seqlen_q` rows, which is what makes the ragged
        # tail stage as zeros instead of faulting. Q rides the forward's K slot
        # and dO its V slot; see the class docstring.
        self.q_slab_span_elems = _slab_span_elems(
            self.seqlen_q_v, self.stride_q_seq, self.hdim_qk
        )
        self.q_slab_base0_bytes = self._slab_byte_base(
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            self.q_row_off,
            head0,
        )
        self.q_head_step_bytes = fx.Index(self.stride_q_head) * elem_bytes

        self.do_slab_span_elems = _slab_span_elems(
            self.seqlen_q_v, self.stride_do_seq, self.hdim_vo
        )
        self.do_slab_base0_bytes = self._slab_byte_base(
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.q_row_off,
            head0,
        )
        self.do_head_step_bytes = fx.Index(self.stride_do_head) * elem_bytes

        # LSE and delta, through **the same function the forward writes LSE
        # with**. Both are always compact -- their strides are a function of
        # VarlenBits, the head count and the token pitch rather than a free
        # variable -- so re-deriving the layout here would be a second source
        # of truth for one fact. `lse_tokens_i32` is
        # `lse_token_pitch`'s answer, which is `max_seqlen_q` for a batched
        # layout and the batch total for a stacked one.
        #
        # The resource is rebased at that `base` and bounded at this sequence's
        # own rows, which is what keeps a q row past `seqlen_q` reading **zero**
        # rather than a neighbouring row's value. That matters more than it
        # looks: a padding row's `P` is `exp2(-lse * log2e)`, and a neighbour's
        # very negative LSE would make it `+inf` -- then `inf * 0` from the
        # zero-staged dO is a NaN in dV, where a zero gives `P = 1` and a clean
        # zero contribution.
        lse_base, self.lse_pitch = fmha.lse_row_addressing(
            self.varlen_bits_arg,
            self.batch_idx,
            head0,
            fx.Index(self.num_head_q),
            fx.Index(self.lse_tokens_i32),
            self.q_row_off,
        )
        self.lse_base0_bytes = lse_base * fx.Index(4)
        self.lse_head_step_bytes = fmha.lse_row_step_per_head(
            self.varlen_bits_arg, fx.Index(self.lse_tokens_i32)
        ) * fx.Index(4)
        self.lse_row_span_bytes = self.seqlen_q_v * self.lse_pitch * fx.Index(4)

        # The bias is `(batch, q head, q row, kv col)`, so it moves with the
        # query head too -- but only its base does. The bound is a function of
        # the two sequence lengths and the row stride, none of which are the
        # head's, so it is computed once here.
        if const_expr(self.traits.BIAS_TYPE):
            self.bias_num_records_bytes = _bias_slab_num_records_bytes(
                self.seqlen_q_v,
                self.seqlen_kv_v,
                self.stride_b_seq_q,
                self.traits.BF16_BYTES,
            )
            self.bias_base0_bytes = self._slab_byte_base(
                self.stride_b_batch,
                self.stride_b_head,
                self.stride_b_seq_q,
                self.q_row_off,
                head0,
            )
            self.bias_head_step_bytes = fx.Index(self.stride_b_head) * elem_bytes

        if const_expr(self.GQA_PACK):
            # The packed axis addresses the whole group from head 0's base, so every descriptor spans `G - 1` more head steps than
            # one head's slab; a packed row past the group is redirected to the oob element each tensor names (its own span).
            extra = self.gqa_group - fx.Index(1)
            self.q_slab_span_elems = self.q_slab_span_elems + extra * fx.Index(
                self.stride_q_head
            )
            self.do_slab_span_elems = self.do_slab_span_elems + extra * fx.Index(
                self.stride_do_head
            )
            self.lse_row_span_bytes = (
                self.lse_row_span_bytes + extra * self.lse_head_step_bytes
            )
            self.lse_head_step_elems = self.lse_head_step_bytes // fx.Index(4)
            self.pack_q_oob = self.q_slab_span_elems
            self.pack_do_oob = self.do_slab_span_elems
            self.pack_lse_oob = self.lse_row_span_bytes
            if const_expr(self.traits.BIAS_TYPE):
                self.bias_num_records_bytes = as_mlir_value(
                    fx.Index(self.bias_num_records_bytes)
                    + extra * self.bias_head_step_bytes
                )
                self.pack_bias_oob = fx.Index(self.bias_num_records_bytes)

    def bind_q_head(self, g):
        """Every descriptor keyed on the query head, in one place.

        **This exists because of GQA**, and it is the whole of the addressing
        side of that feature: several query heads reduce into one KV head, so a
        workgroup walks the group and re-points the query side at each head in
        turn while K, V, dK and dV stay exactly where they are. Collected into
        one method rather than left inline so that the loop and the prologue
        cannot drift -- there is one description of what "the query side" is.

        `g` is the group-relative head index, an `Index`: 0 from the prologue in
        `init_descriptors`, the loop variable from `retarget_q_head`. Each
        descriptor is its head-0 base plus `g` head steps rather than a fresh
        derivation from the strides -- see `_init_q_head_invariants` for what
        that difference buys, which is scalar pressure and only that. At
        `num_kv_heads == num_heads` the group is one head wide, the loop runs
        once and `g` is 0.
        """
        self.q_head_idx = self.gqa_q_head_base + g

        self.k_div = self._slab_view_at(
            self.Q,
            self.q_slab_base0_bytes + g * self.q_head_step_bytes,
            self.q_slab_span_elems,
        )
        self.v_div = self._slab_view_at(
            self.DO,
            self.do_slab_base0_bytes + g * self.do_head_step_bytes,
            self.do_slab_span_elems,
        )
        self.q_div = self.k_div

        base_bytes = self.lse_base0_bytes + g * self.lse_head_step_bytes
        self.lse_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))),
            base_bytes,
            self.lse_row_span_bytes,
        )
        self.delta_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.Delta))),
            base_bytes,
            self.lse_row_span_bytes,
        )

        if const_expr(self.traits.BIAS_TYPE):
            self.bias_rsrc = buffer_ops.create_buffer_resource(
                self.Bias,
                max_size=False,
                num_records_bytes=self.bias_num_records_bytes,
                base_byte_offset=as_mlir_value(
                    self.bias_base0_bytes + g * self.bias_head_step_bytes
                ),
            )

        self._bind_philox_plane()

    def _bind_philox_plane(self):
        """This query head's philox plane.

        The plane is `(sequence, q head)`, so a GQA group draws a *different*
        mask per head -- which is the forward's behaviour, since the forward has
        one program per query head and this must reproduce it bit for bit.
        `seq_idx_i32` and not `batch_idx` for the same reason the forward uses
        it: the decode collapses every stacked sequence to batch 0, which would
        hand a whole packed batch one plane.

        Guarded on the attribute rather than on the trait because `init_philox`
        runs *after* `init_descriptors`: the prologue's `bind_q_head` finds no
        RNG and does nothing, `init_philox` then calls this directly, and the
        loop's calls find one.
        """
        if const_expr(not self.traits.ENABLE_DROPOUT):
            return
        if getattr(self, "philox_rng", None) is None:
            return
        plane = self.seq_idx_i32 * fx.Int32(self.num_head_q) + fx.Int32(self.q_head_idx)
        self.philox_plane_base, self.philox_row_stride = self.philox_rng.grid_plane(
            self.philox_offset_base_v, plane, self.seq_len_v, self.seq_len_kv_v
        )

    def retarget_q_head(self, g):
        """Point the query side at head `g` of this KV head's group.

        `g` is a runtime `Index`. The KV side -- K, V, dK, dV, and the
        accumulators they feed -- is untouched by construction: nothing below
        `bind_q_head` reads `q_head_idx`.
        """
        self.bind_q_head(g)

    def _slab_rsrc(self, tensor, s0, s1, s2, hdim, pin=False):
        """A raw buffer resource over this workgroup's KV slab, bounded at its rows.

        Only the four KV-side tensors need one, so the head, the row origin and
        the batch index are the KV ones rather than parameters -- which is also
        what stops Q's batch index reaching them under `0x040B`.

        Bounded through `_slab_span_elems`, the same as the `_slab_view` over
        the same slab: these are two access shapes over one region, and a
        region cannot have two ends.
        """
        span_bytes = _slab_span_elems(self.seqlen_kv_v, s2, hdim) * fx.Index(
            self.traits.BF16_BYTES
        )
        if const_expr(pin):
            addr_i64 = as_mlir_value(
                fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
                + fx.Int64(
                    self._slab_byte_base(
                        s0,
                        s1,
                        s2,
                        self.kv_row_off,
                        self.kv_head_idx,
                        batch_idx=self.kv_batch_idx,
                    )
                )
            )
            return buffer_ops.create_buffer_resource_from_addr(
                _pin_sgpr(addr_i64),
                num_records_bytes=_pin_sgpr(as_mlir_value(fx.Int64(span_bytes))),
            )
        return dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(tensor))),
            self._slab_byte_base(
                s0,
                s1,
                s2,
                self.kv_row_off,
                self.kv_head_idx,
                batch_idx=self.kv_batch_idx,
            ),
            span_bytes,
        )

    # -- bounds ------------------------------------------------------------

    def _init_pack_rows(self):
        """`gqa_pack`: the constants `m16.pack_split` reads. `pack_rows` is the packed axis's extent, `Sq * G`."""
        sq = fx.Index(
            (self.seqlen_q_v > fx.Index(0)).select(self.seqlen_q_v, fx.Index(1))
        )
        self.pack_sq = sq
        self.pack_rows = self.seqlen_q_v * self.gqa_group
        self.pack_magic = fx.Int64(fx.Index(0x7FFFFFFF) // sq + fx.Index(1))

    def init_tile_bounds(self, **kwargs):
        """The q tiles this KV block walks, rounded out to an even count.

        The body consumes two tiles per iteration, so an odd count would leave
        the second half of the last iteration reading a tile the walk did not
        intend. Rounding up costs one tile and is safe in both modes: dense,
        because rows past `seqlen_q` stage as zeros; causal, because the
        per-tile predicate below is a *superset* test and masks a tile outside
        the live range to nothing.

        **Under causal the walk does not start at tile 0**, and that is the
        whole of the tile cut. A literal-zero tile base in the prologue or the
        loop would leave the cut inert; `split_t0` is the one knob both read, so
        there is one place to get it right.
        """
        traits = self.traits
        if const_expr(not traits.CAUSAL):
            self.window_left_i32 = None
            self.causal_lo_i32 = None
            rows_q = self.seqlen_q_v
            if const_expr(self.GQA_PACK):
                # One pass over the group's heads laid end to end; see `m16.pack_split`.
                rows_q = rows_q * self.gqa_group
                self._init_pack_rows()
            tiles = (rows_q + fx.Index(traits.BLOCK_Q - 1)) // fx.Index(traits.BLOCK_Q)
            tiles = ((tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
            tiles = fx.Index((tiles < fx.Index(2)).select(fx.Index(2), tiles))
            self.split_t0 = fx.Index(0)
            self.split_t_end = tiles
        else:
            # **The sentinels resolve on the device**, against this sequence's
            # own lengths, because under varlen they differ per sequence.
            # A non-window build forwards `WINDOW_BOTRIGHT` for both, so the
            # left bound comes back as `seqlen_q` -- unbounded -- and every
            # left-bound comparison below is `const_expr`-ed away regardless.
            left, right = fmha.resolve_window(
                self.window_left_arg,
                self.window_right_arg,
                fx.Int32(self.seqlen_q_v),
                self.seqlen_kv_i32,
            )
            self.window_left_i32 = fx.Int32(left)
            # A (q, kv) element is live iff `kv - right <= q <= kv + left`, so
            # what the causal side compares against is `-right`. Stored negated
            # because that is the form every element test wants.
            self.causal_lo_i32 = fx.Int32(fx.Int32(0) - fx.Int32(right))
            # **The same function the forward cuts a Q block's KV range with,
            # axes swapped**: a KV block's Q range. gfx1201's dK/dV established
            # the identity and the contract says not to re-derive it. Row axis
            # is KV here, column axis is Q, so `window_left` takes the resolved
            # *right* bound and vice versa.
            regions = fmha.decompose_causal_regions(
                fx.Int32(self.kv_start),
                self.seqlen_kv_i32,
                fx.Int32(self.seqlen_q_v),
                fx.Int32(right),
                fx.Int32(left),
                traits.BLOCK_KV,
                traits.BLOCK_Q,
                fx.Boolean(True),
            )
            n_tiles = regions.n_left + regions.n_full + regions.n_right
            n_tiles = fx.Index((n_tiles > fx.Int32(0)).select(n_tiles, fx.Int32(0)))
            if const_expr(self.GQA_PACK):
                # Every head of the group has the same live q range `[lo_q, hi_q)`, so the packed walk spans head 0's first live
                # row to the last head's last live row. Dead gaps between heads inside it are walked and masked to nothing.
                self._init_pack_rows()
                lo_q = fx.Index(regions.left_col0)
                hi_q = fx.Index(
                    (
                        fx.Index(n_tiles * fx.Index(traits.BLOCK_Q) + lo_q)
                        < self.seqlen_q_v
                    ).select(n_tiles * fx.Index(traits.BLOCK_Q) + lo_q, self.seqlen_q_v)
                )
                end_row = (self.gqa_group - fx.Index(1)) * self.seqlen_q_v + hi_q
                end_tile = (end_row + fx.Index(traits.BLOCK_Q - 1)) // fx.Index(
                    traits.BLOCK_Q
                )
                n_tiles = fx.Index(
                    (n_tiles > fx.Index(0)).select(
                        end_tile - lo_q // fx.Index(traits.BLOCK_Q), fx.Index(0)
                    )
                )
            n_tiles = ((n_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
            self.split_t0 = fx.Index(regions.left_col0) // fx.Index(traits.BLOCK_Q)
            self.split_t_end = self.split_t0 + n_tiles
        self.num_q_tiles = self.split_t_end - self.split_t0
        self.max_num_tiles = self.num_q_tiles
        self.causal_end_raw_i32 = None


class BwdDkDvStreamLoader(ParityKvGmemToLdsLoader):
    """Stages Q and dO into the K and V LDS slots.

    Two wrappers rather than an override of `load_k` / `load_v`, because the
    parity loader's job -- swapping in the right per-tensor sequence stride
    before delegating -- is exactly what has to change, and the tensor it must
    swap in is not the one the method is named for. Calling the *production*
    method directly keeps one implementation of the DMA sequence and states the
    substitution at the call site instead of hiding it in an attribute.

    **The descriptor is re-read from `ctx_ref` every call, and under GQA that
    is load-bearing.** `DualwaveKernelContext.__init__` does
    `self.__dict__.update(ctx.__dict__)`, so a helper holds a *snapshot* of the
    context taken when it was constructed -- before the group loop exists.
    `retarget_q_head` rebinds `ctx.k_div` to the next query head's slab, and a
    loader reading its own `self.k_div` would keep staging the first head's Q
    for the whole group: finite, plausible, and wrong by exactly the group
    sum. Everything else the DMA path touches (`k_dma_m0`, the tile start, the
    lane offsets) is head-independent, so these two lines are the whole fix.
    """

    def stage_q_tile(self, tile_idx, buf_id):
        ctx = self.ctx_ref
        self.stride_kv_n_v = ctx.stride_q_seq_v
        self.k_div = ctx.k_div
        self.dma_stage = 0
        if const_expr(self.GQA_PACK):
            self._pack_begin(
                tile_idx, ctx.stride_q_head, ctx.stride_q_seq_v, ctx.pack_q_oob
            )
        dualwave.DualwaveKvGmemToLdsLoader.load_k(
            self, self.tile_start(tile_idx), buf_id
        )

    def stage_do_tile(self, tile_idx, buf_id):
        ctx = self.ctx_ref
        self.stride_kv_n_v = ctx.stride_do_seq_v
        self.v_div = ctx.v_div
        self.dma_stage = 0
        if const_expr(self.GQA_PACK):
            self._pack_begin(
                tile_idx, ctx.stride_do_head, ctx.stride_do_seq_v, ctx.pack_do_oob
            )
        dualwave.DualwaveKvGmemToLdsLoader.load_v(
            self, self.tile_start(tile_idx), buf_id
        )

    # -- gqa_pack ----------------------------------------------------------

    def _pack_begin(self, tile_idx, stride_head, stride_seq, oob):
        """Stage the next tile from the packed row axis: rows `tile_idx * BLOCK_Q + n` map to `(head, q)` per lane.

        The tile's row origin rides in each lane's element offset (see `kv_src_elem`), so the DMA's scalar offset is zero.
        """
        self._pack_row0 = self.tile_start(tile_idx)
        self._pack_head = fx.Index(stride_head)
        self._pack_seq = stride_seq
        self._pack_oob = oob
        self._pack_rows = {}
        self.stride_kv_n_v = fx.Index(0)

    def kv_src_elem(self, src_base, d_flat):
        if const_expr(not self.GQA_PACK):
            return super().kv_src_elem(src_base, d_flat)
        band, issue = self._issue_split(d_flat)
        if issue not in self._pack_rows:
            n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + (
                self.wave_id + issue * self.traits.NUM_WAVES
            )
            g, q, valid = m16.pack_split(self, self._pack_row0 + n_in_tile)
            self._pack_rows[issue] = fx.Index(
                valid.select(g * self._pack_head + q * self._pack_seq, self._pack_oob)
            )
        return (
            self._pack_rows[issue]
            + self.d_bucket * self.traits.VEC_KV
            + band * self.traits.D_128B_SIZE
        )


class BwdDkDvResidentLoader(dualwave.DualwaveKernelContext):
    """K and V rows for this wave, loaded once and held for the whole q loop.

    The same shape of read as `DualwaveQLoader.load_pack` -- 128 bits per lane
    at `row * stride + ks * K_STEP_QK + (lane // 32) * MFMA_LANE_K` -- because
    the MFMA's A and B operands take identical per-lane layouts, so what the
    forward builds for Q is what this needs for K and V. Kept as a list of
    per-`ks` packs rather than one concatenated vector: the forward
    concatenates so `_get_q_pack` can slice a loop-invariant register, and here
    the packs feed two different GEMMs and never need to be adjacent.

    **This is where a padded head is masked**, and it is the cheap side of the
    trade. K and V are read once per kernel, so masking every k-step here costs
    nothing measurable -- against the forward, which has to mask its streamed
    K on the hot path and measured 27-54% for it.
    """

    def load_resident(self, which, hdim):
        traits = self.traits
        div = self.k_res_div if which == "k" else self.v_res_div
        elem_base = self.k_res_elem_base if which == "k" else self.v_res_elem_base
        stride_seq = self.stride_k_seq_v if which == "k" else self.stride_v_seq_v
        row_in_block = self.kv_row_in_block
        cols = None
        if const_expr(self.PADDED_HEAD):
            # Loop-invariant and dead immediately after the prologue, so the
            # bitmask form is pure win here for the same reason it is in the
            # forward's Q loader.
            cols = MaskedAxis(fx.Index(hdim), elem_dtype=self.elem_dtype, bitmask=True)
        packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            col = fx.Index(ks * traits.K_STEP_QK) + self.lane_div_32 * fx.Index(
                traits.MFMA_LANE_K
            )
            raw = dualwave._buffer_load_128(
                elem_base + row_in_block * stride_seq + col,
                _load_atom_128=self.load_atom_128,
                q_div=div,
                q_load_i32x4_type=self.q_load_i32x4_type,
            )
            pack = Vec(raw, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value()
            if const_expr(self.PADDED_HEAD):
                pack = cols.discard(pack, col, traits.MFMA_LANE_K)
            packs.append(pack)
        return packs


class BwdDkDvStreamReader(dualwave.DualwaveKernelContext):
    """The two ways one staged tile is read back.

    Row-major for the d-contracted GEMMs and column-major for the q-contracted
    ones, from the *same* LDS bytes. That is the architectural bet of plan
    section 1 -- two LDS tiles, not gfx1201's four -- and it holds because
    `ds_read_b64_tr_b16` was designed for exactly this operand.
    """

    # Set by the kernel once the floor is known; see `_masked_ks_steps`.
    masked_steps = ()

    def read_row_pack(self, tile_base, scope_name, half, ks, hdim):
        """One `A[row][d]` pack for k-step `ks` of rows `half * 32 .. + 32`.

        `K_LDS_TO_REG_N_STRIP_STRIDE` is 32 tokens along the intra-line token
        axis, so the two halves are the same columns of consecutive row blocks
        -- the relationship the forward's `_load_k_pair` has between the halves
        of a KV tile, hoisted into a parameter.

        **One pack, not a whole half.** The caller feeds each straight into its
        MFMA, so nothing outlives its use; reading the half up front would put
        `K_STEPS_QK * 4` VGPRs live at once, which is 128 at head_dim 512
        beside accumulators that have already spent most of the file. Splitting
        the pair into halves was worth 2.2x at head_dim 128 before the wave
        count was touched at all, and this is the same move one step further.
        """
        traits = self.traits
        idx = (
            tile_base
            + self.stream_row_read_base
            + half * traits.K_LDS_TO_REG_N_STRIP_STRIDE
            + _stream_ks_offset(traits, ks)
        )
        pack = _lds_pack_read(
            traits, self.lds_kv_base_ptr, idx, scope_name, self.kv_mfma_pack_type
        )
        if const_expr(self.PADDED_HEAD) and ks in self.masked_steps:
            # The bitmask form ANDs one precomputed dword per pair instead of
            # selecting per element, but each mask stays live for the whole q
            # loop. Gated on how many steps actually survive the floor rather
            # than on `K_STEPS_QK`: the forward measured +21% in its 64-wide
            # tile and -43% in the 128-wide one, where 32 extra live registers
            # turned a spill-free build into 61 spills.
            width = traits.MFMA_LANE_K
            col = fx.Index(ks * traits.K_STEP_QK) + self.lane_div_32 * fx.Index(width)
            cols = MaskedAxis(
                fx.Index(hdim),
                elem_dtype=self.elem_dtype,
                bitmask=len(self.masked_steps) * (width // 2) <= 16,
            )
            pack = cols.discard(pack, col, width)
        return pack

    def read_column_chunk(self, tile_base, scope_name, dc):
        """The four `A[d][row]` packs for this shard's output chunk `dc`.

        `dc` is **shard-local**: the shard's first chunk is folded into
        `stream_col_read_base` (see `init_lds_read_bases`), because the
        immediate offset `_v_imm_lo` builds has to be a compile-time constant
        and the shard index is not.

        One chunk at a time, not the forward's whole `[4][D_CHUNKS]` table: the
        transposed operand feeds an accumulator that is already 64 VGPRs at
        head_dim 128, and reading all of it up front puts 128 more beside it
        for no benefit -- each pack is consumed by one MFMA and dies.
        """
        traits = self.traits
        lds_base = tile_base + self.stream_col_read_base
        pair = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
        packs = []
        for k_substep in range_constexpr(4):
            imm_lo = _v_imm_lo(traits, dc, k_substep)
            a = _ds_read_tr_v4f16_imm(
                lds_base,
                imm_lo,
                lds_kv_base_idx=self.lds_kv_base_idx,
                v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            b = _ds_read_tr_v4f16_imm(
                lds_base,
                imm_lo + pair,
                lds_kv_base_idx=self.lds_kv_base_idx,
                v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            packs.append(Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class BwdDkDvGemmHelper(ParityGemmHelper):
    """The four GEMMs, as two shapes.

    Both are `sum_k mfma(A_k, B_k, acc)`; they differ only in what indexes `k`
    and whether the accumulator is fresh. Naming them for the axis they
    contract rather than for the tensors keeps `S` and `dP` -- and `dV` and
    `dK` -- one piece of code each instead of two, which is what stops a
    transcription drifting.
    """

    def contract_d(self, read_a, b_packs):
        """`S` or `dP`: reduce over the head dim into a fresh accumulator.

        `read_a` is a *callable*, not a list, so each A pack is read into the
        MFMA that consumes it rather than the whole set being gathered first.
        See `BwdDkDvStreamReader.read_row_pack`.
        """
        acc = self.c_zero_v16f32
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            acc = dualwave._mfma_acc(
                read_a(ks), b_packs[ks], acc, self.mma_atom, self.mfma_acc_vec_type
            )
        return acc

    def contract_q(self, a_packs, b_packs, acc):
        """`dV` or `dK`: reduce over the tile's 64 q rows into a carried accumulator.

        Four k-substeps of 16 rows. `a_packs[step]` and `b_packs[step]` must
        carry the *same* permutation of the q axis within a substep, and they
        do: the transpose read's element order is `[0,1,2,3,8,9,10,11]` and
        `_pack_p_v8_slices` slices the accumulator in exactly that order. The
        forward relies on the identical coincidence between its V read and its
        P packing, one axis over.
        """
        for step in range_constexpr(4):
            acc = dualwave._mfma_acc(
                a_packs[step], b_packs[step], acc, self.mma_atom, self.mfma_acc_vec_type
            )
        return acc


class BwdDkDvSoftmaxHelper(ParitySoftmaxHelper):
    """`P` and `dS` from the two score accumulators, plus their row constants.

    Nothing online here: the forward already ran the softmax and its `LSE` is
    the whole of the state this needs, which is why the backward has no running
    max, no rescale and no `l`. What it does have that the forward does not is
    a **per-element** row index -- the accumulator's 16 f32 are 16 different q
    rows of one kv row, where the forward's are 16 kv columns of one q row -- so
    `lse` and `delta` arrive as 16 registers rather than one.
    """

    def load_row_values(self, rsrc, tile_base, half, scale):
        """16 f32, one per accumulator element of `half`, times `scale`.

        The row an element holds is `8 * (r // 4) + 4 * (lane // 32) + (r % 4)`,
        so the four `r % 4` values of a group are four *contiguous* rows and a
        dense build covers each group with one `buffer_load_dwordx4`.
        `_ROW_RUNS` is where that grouping is stated.

        **Under the `_TH` logsumexp layout the four rows are `num_heads`
        apart**, so the wide load does not apply and it is sixteen scalars
        instead.

        **The choice is made at runtime, and it has to be.** `lse_row_addressing`
        decodes the layout from `VarlenBits` at *runtime*, so `base` and `pitch`
        are right for either layout; a build that fixed the arm at build time
        and was handed the other layout's bits read the wrong elements,
        silently, and a host-side wrapper is no defence because AOTriton's C++
        launcher never calls it. Measured by bypassing such a wrapper, packed
        varlen, two sequences, head_dim 64 (columns are the build's layout and
        the bits' layout):

            num_head_q    build=HT/bits=HT   build=TH/bits=TH   build=HT/bits=TH
                 4            2.36e-03           2.36e-03          3.34e-01
                 8            2.36e-03           2.36e-03          3.33e-01
                 1            2.35e-03           2.35e-03          2.35e-03

        -- the same defect shape as the GQA trip count, and unreachable at
        `num_head_q == 1`, where the two layouts coincide exactly
        (`base_ht == base_th`, both pitches 1). AOTriton pins `num_heads=1` at
        build but passes the real count as a kernarg, which is what made this
        reachable.

        **The predicate is `pitch != 1`, not the bits.** It comes from the one
        decode rather than a second reading of `VarlenBits`, so the two cannot
        drift -- and it is the sharper test: the wide load is valid exactly
        when the pitch is 1, which includes TH at `num_head_q == 1`.

        **The branch is outside the tile loop, and it has to be.** `LSE_STRIDED`
        is a *Python* bool here: the kernel body is traced twice, once per arm,
        and one runtime `scf.if` on `ctx.lse_strided` picks a whole body. Put
        the branch here instead -- an `scf.if` per call, four per tile -- and
        the loop pays for it whichever arm it takes, because the region is a
        scheduling barrier the row loads can no longer be hoisted across.
        Measured, `B=2 H=8 S=4096`, branch-inside against this:

            head_dim        32     64     96    128    224    512
            branch inside  450    497    603    700    355    433
            branch outside 494    741    765    812    627    430

        -- the inside form performs as though it always took the scalar arm,
        which is the 0.68x that made this a build axis in the first place. The
        cost of the outside form is that the body is emitted twice.

        **The strided arm's address is one divergent value plus sixteen
        uniform ones, and writing it the other way cost a third of the
        kernel.** `(row_base + T[r]) * pitch` and `row_base * pitch +
        T[r] * pitch` compute the same offset, but the first makes all sixteen
        *divergent*: sixteen `v_mul_lo_u32` and sixteen VGPRs that must be live
        together, because the loads are issued back to back to overlap their
        latency. The second multiplies once per call in a VGPR and once per
        `r` in an **SGPR** -- `pitch` is decoded from `VarlenBits` and the head
        count, both workgroup-uniform, and `_ROW_THRESHOLDS` is a Python
        constant -- so what reaches the loads is one base plus a scalar.

        Fifteen VGPRs is not a rounding error here: it is the 256 boundary. A
        wave that fits in 256 gets two per SIMD on this geometry -- the LDS
        admits two workgroups per CU, so two is what the geometry is for -- and
        a wave that needs 260 gets one. Nothing says so out loud, because
        `waves_per_eu` is 1 at this rung and LLVM therefore budgets 512 and has
        no reason to stop at 256. Measured, `B=2 H=8 S=4096`, head_dim 256
        causal, `mfma_rows=16 block_q=32`:

            arms              VGPRs   waves/SIMD   TFLOP/s
            wide only           239        2         1403
            strided only        259        1          786
            both                268        1          783
            both, this form     253        2         1392

        -- 0.56x from fifteen registers, with nothing else about the loop
        changed, and the dense build's 239 sitting seventeen short of the cliff
        the whole time. That is the entire varlen dK/dV regression, at four
        rungs; `_FEATURE_OVERRIDES` records the three whose table entries had
        to be put back afterwards.

        **`soffset` is not available for the uniform term**, tempting as it
        looks: CDNA4 9.1.5.1 range-checks a raw buffer on
        `InstOffset + vgpr_offset` alone, so an offset carried in `soffset`
        addresses memory without being bounded. The bound is what makes a q row
        past `seqlen_q` read zero rather than a neighbour's very negative LSE,
        and `init_descriptors` explains what that zero is worth. The uniform
        term therefore rides a `v_add` into the voffset, which is checked.
        """
        values = [None] * 16
        row_base = fx.Int32(tile_base + fx.Index(half * 32)) + fx.Int32(
            self.lane_div_32
        ) * fx.Int32(4)
        if const_expr(not self.LSE_STRIDED):
            for elem0, col_off, width in _ROW_RUNS:
                span = buffer_ops.buffer_load(
                    rsrc,
                    as_mlir_value(row_base + fx.Int32(col_off)),
                    vec_width=width,
                    dtype=fx.Float32,
                )
                vec = Vec(span, (width,), fx.Float32)
                for j in range_constexpr(width):
                    values[elem0 + j] = _fmul(vec[j], scale, self.fm_fast)
            return values
        pitch = self.lse_pitch
        # One divergent multiply, sixteen uniform ones. Not `(row_base +
        # T[r]) * pitch`; see the docstring for the twenty VGPRs that costs.
        row_v = fx.Index(row_base) * pitch
        for r in range_constexpr(16):
            off = row_v + fx.Index(_ROW_THRESHOLDS[r]) * pitch
            one = buffer_ops.buffer_load(
                rsrc, as_mlir_value(fx.Int32(off)), vec_width=1, dtype=fx.Float32
            )
            values[r] = _fmul(fx.Float32(one), scale, self.fm_fast)
        return values

    def probabilities(self, v_s, neg_lse2, bias2=None):
        """`P = exp2(qk_scale * S + bias*log2e - log2(e) * LSE)`, element by element.

        Q is **not** pre-scaled, which is the one place this departs from the
        forward, and the cost of getting it wrong is measured: folding
        `sm_scale * log2e` into Q and rounding to bf16 puts the error in the
        exponent, taking the error ratio from 1.29 at `sm_scale = 0.05` to 10.9
        at 1.0. `O` is a normalised average and absorbs it; `dS` is not and does
        not. Here the scale rides the subtraction that had to happen anyway, so
        it is also free.

        **The bias goes after the scale**, which is the forward's rule read in
        this kernel's terms. There it means "after the pre-scale that already
        happened upstream"; here Q is unscaled, so it means after the explicit
        `c_sm_scale_log2e` multiply on this line. Either way the bias is in
        natural units and the exponent is base-2, so the conversion is the
        thing that has to line up, not the position in the expression.

        The whole chain drops to `fm_bias` in a bias build so a `-inf` bias
        survives to `exp2`, which returns an exact zero for it -- and a zero `P`
        then kills every downstream term, since `dS = P*(...)` and
        `dV += P.dO`. `BIAS_TYPE` is a build axis, so a build without bias
        emits the identical `fm_fast` chain it always did.
        """
        values = [Vec(v_s)[r] for r in range_constexpr(16)]
        scale = self.ctx_ref.c_sm_scale_log2e
        fm = self.fm_fast
        if const_expr(bias2 is not None):
            fm = fx.arith.FastMathFlags.contract | fx.arith.FastMathFlags.reassoc
        scaled = [_fmul(values[r], scale, fm) for r in range_constexpr(16)]
        if const_expr(bias2 is not None):
            scaled = [_fadd(scaled[r], bias2[r], fm) for r in range_constexpr(16)]
        return exp2_wait_state(
            [
                dualwave.rocdl.exp2(
                    T.f32, as_mlir_value(_fadd(scaled[r], neg_lse2[r], fm))
                )
                for r in range_constexpr(16)
            ]
        )

    # -- the forward's bias path, shadowed ---------------------------------
    #
    # `ParitySoftmaxHelper` carries `_add_bias_inplace`, `bias_to_lists` and a
    # `seq_pad_mask_if_needed` override, and all three address the bias as
    # *one q row per lane, 16 KV columns per element* -- the transpose of this
    # accumulator. Nothing in the dK/dV body calls them today, so inheriting
    # them is dead code rather than a wrong answer; the hazard is that they are
    # dead and *plausible*, and the next person to wire a bias site would get
    # finite garbage from a method that looks like the one to call.
    #
    # Shadowed rather than deleted, because deleting them would mean editing
    # the forward's helper.

    def _add_bias_inplace(self, v_s, tile_idx):
        raise NotImplementedError(
            "the forward's bias addressing does not transpose: it assumes one q row per lane and "
            "contiguous KV columns per element, where this accumulator holds one KV column per lane "
            "and 16 q rows. Use `bias_log2e(ctx, q_rows)` and `probabilities(..., bias2=)`."
        )

    def bias_to_lists(self, v_s, tile_idx):
        raise NotImplementedError(
            "dK/dV applies the bias in `probabilities`; see `_add_bias_inplace`"
        )

    def seq_pad_mask_if_needed(self, v_s, tile_idx):
        raise NotImplementedError(
            "dK/dV has no KV tail mask on the scores: the KV extent is a descriptor bound, not a "
            "select, and the q tail is handled by the same means. See `_add_bias_inplace`."
        )

    def dscores(self, p_list, v_dp, delta, keep=None):
        """`dS = P * (dP - delta)`, with dropout's chain rule folded into `dP`.

        `dP` is deliberately unscaled by `sm_scale`: it is `dO . V^T` exactly,
        and `sm_scale` belongs to `dK` alone -- `dS` is the gradient with
        respect to the *scaled* logits, which is what `P` was built from.

        **Under dropout `dP` picks up `keep` and the survivor scale, and `P`
        does not**, which is the one place the two masks do not compose. The
        forward computes `O = sum_j keep*P*s*V`, so differentiating gives
        `dS = P * (keep*s*dP - delta)` -- with the *undropped* `P` outside the
        bracket, because the softmax denominator is the undropped sum. Reusing
        the dropped `P` there (as the causal mask lets you) drops the
        `-P*delta` term wherever a survivor was dropped, which is finite,
        plausible and wrong.
        """
        values = [Vec(v_dp)[r] for r in range_constexpr(16)]
        if const_expr(keep is not None):
            s = self.ctx_ref.c_dropout_scale
            values = [
                keep[r].select(_fmul(values[r], s, self.fm_fast), self.c_zero_f)
                for r in range_constexpr(16)
            ]
        return [
            _fmul(p_list[r], _fsub(values[r], delta[r], self.fm_fast), self.fm_fast)
            for r in range_constexpr(16)
        ]

    def q_rows(self, tile_idx, half):
        """The absolute q row each of this half's 16 accumulator elements holds.

        `8*(r//4) + 4*(lane//32) + (r%4)`, which `_ROW_THRESHOLDS` states once
        and `load_row_values`, `mask_if_clipped` and the dropout mask all read.
        """
        base = (
            tile_idx * fx.Index(self.traits.BLOCK_Q)
            + fx.Index(half * 32)
            + self.lane_div_32 * fx.Index(4)
        )
        return [base + fx.Index(_ROW_THRESHOLDS[r]) for r in range_constexpr(16)]

    def mask_if_clipped(self, p_list, tile_idx, half):
        """Zero `P` outside the band, for the tiles that can be clipped.

        **The mask goes on `P`, not on `S`, and that is both cheaper and
        safer.** `dS = P * (dP - delta)` and `dV += P . dO`, so a zero in `P`
        kills every downstream contribution -- one select per element covers
        both gradients, where the forward has to mask `S` because its `O`
        depends on the softmax denominator. It also keeps `-inf` out of the
        arithmetic entirely: `fm_fast` carries `ninf`, and plan1 records that
        licence deleting a KV tail mask on gfx1201.

        The predicate is **wave-uniform** -- it is built from the wave's own KV
        row range and the tile index, both `readfirstlane`-derived -- and it is
        a *superset* test: any tile that needs any masking takes the branch,
        including one entirely outside the live range, which is what makes the
        even-rounded tile count safe.

        Masking unconditionally would also be correct. The forward measured
        that at 197 us against causal's 126, because it forces the mask onto
        every interior tile in the walk.

        **No transpose read is inside this region**, and that is a correctness
        requirement rather than tidiness (CDNA4 section 11.4: EXEC must be all
        1s across `ds_read_b64_tr_b16`). The q-contracted GEMMs run in their own
        loop after the softmax, so a narrowed EXEC never reaches a transpose read.
        """
        traits = self.traits
        ctx = self.ctx_ref
        lo_i32 = ctx.causal_lo_i32
        left_i32 = ctx.window_left_i32
        rows = traits.MFMA_ROWS
        # Per-lane, per-tile: `q - kv` for accumulator element 0 of this half.
        rel0 = (
            fx.Int32(tile_idx * fx.Index(traits.BLOCK_Q))
            + fx.Int32(half * 32)
            + fx.Int32(self.lane_div_32) * fx.Int32(4)
            - fx.Int32(ctx.kv_row)
        )
        # Wave-uniform bounds on the same quantity, over the whole tile and the
        # wave's whole KV row block.
        kv_lo = fx.Int32(ctx.kv_start) + fx.Int32(ctx.wave_kv_offset_uni)
        tile_q0 = fx.Int32(tile_idx * fx.Index(traits.BLOCK_Q))
        # `q >= kv - right` can fail somewhere in this tile iff the tile's
        # first q is below the *largest* kv row's bound.
        need = tile_q0 < kv_lo + fx.Int32(rows - 1) + lo_i32
        if const_expr(traits.WINDOW):
            # `q <= kv + left` can fail iff the tile's last q is above the
            # *smallest* kv row's bound.
            need = need | (tile_q0 + fx.Int32(traits.BLOCK_Q - 1) > kv_lo + left_i32)
        zero_f = self.c_zero_f
        window = traits.WINDOW

        def _apply(vals):
            out = list(vals)
            for r in range_constexpr(16):
                rel = rel0 + fx.Int32(_ROW_THRESHOLDS[r])
                keep = rel >= lo_i32
                if const_expr(window):
                    keep = keep & (rel <= left_i32)
                out[r] = keep.select(fx.Float32(vals[r]), zero_f)
            return out

        @flyc.jit
        def _mask_if_needed(p_vec):
            out = p_vec
            if need:
                out = Vec.from_elements(
                    [
                        as_mlir_value(v)
                        for v in _apply([Vec(p_vec)[r] for r in range_constexpr(16)])
                    ],
                    fx.Float32,
                ).ir_value()
            return out

        packed = Vec.from_elements(
            [as_mlir_value(fx.Float32(v)) for v in p_list], fx.Float32
        ).ir_value()
        return [Vec(_mask_if_needed(packed))[r] for r in range_constexpr(16)]

    def pack_half(self, values):
        """The two bf16 B-operand packs for one 32-row half of a q tile.

        `_pack_p_v8_slices`' inner slicing, done a half at a time. The forward
        packs both halves together because its two accumulators are live
        together anyway; here they deliberately are not (see
        `BwdDkDvTileBody.run`), and a whole-tile packer would force them to be.

        Each pack leaves through `mfma_operand_wait_state`, which is where the
        `v_cvt_pk_bf16_f32`-into-`v_mfma` wait state comes from; that docstring
        has the measurement. The barrier is on the pack rather than inside
        `contract_q` so that it sits at the producer, which is the end the
        hazard is counted from.
        """
        return [
            mfma_operand_wait_state(
                dualwave._bf16_trunc_pack_v8(
                    self.traits,
                    [values[p * 8 + s] for s in range_constexpr(8)],
                    elem_dtype=self.elem_dtype,
                )
            )
            for p in range_constexpr(self.traits.PV_K_STEPS)
        ]


class BwdDkDvStoreHelper(dualwave.DualwaveStoreHelper):
    """dK and dV, through the forward's O store path.

    A dK/dV accumulator is `[d][kv_row]` -- the same shape as the forward's
    `O` accumulator, one axis renamed -- so the 128-bit packing, the
    `permlane32_swap` half-wave exchange and the store are reused verbatim.
    The descriptor, the row stride, the shard's column origin and the head
    extent are parameters, because two output tensors go through one helper.
    """

    def store_accs(self, accs, which, hdim):
        """Store this wave's `D_CHUNKS_PER_SHARD` chunks of one output.

        **Address the chunk from its global column, not its local one.** Under
        sharding `dc` restarts at 0 while the columns it addresses do not, and
        a padded-head suppression derived from the local index goes wrong: the descriptor spans the
        whole tensor rather than one row, so a store past the row pitch lands
        in the *next row* and corrupts it -- head_dim 300 came out at 0.58
        absolute error, finite and with no fault.
        """
        traits = self.traits
        div = self.dk_div if which == "dk" else self.dv_div
        stride_seq = self.stride_dk_seq_v if which == "dk" else self.stride_dv_seq_v
        oob_off = self.dk_oob_off if which == "dk" else self.dv_oob_off
        base = (
            self.kv_row * stride_seq
            + self.dkv_col_base
            + self.lane_div_32 * fx.Index(8)
        )
        cols = MaskedAxis(fx.Index(hdim)) if const_expr(self.PADDED_HEAD) else None
        for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
            for g in range_constexpr(2):
                off = base + fx.Index(dc * traits.D_CHUNK + 2 * g * 8)
                if const_expr(self.PADDED_HEAD):
                    # A 128-bit store is all-or-nothing, so a chunk straddling
                    # `hdim` writes into the caller's own 8-element pad, which
                    # the input contract permits. Only a chunk *starting* at or
                    # past `hdim` must be dropped, and pushing its offset past
                    # `num_records` is what drops it -- one select on a
                    # lane-varying value instead of an `scf.if` around a store.
                    col = (
                        self.dkv_col_base
                        + fx.Index(dc * traits.D_CHUNK + 2 * g * 8)
                        + self.lane_div_32 * fx.Index(8)
                    )
                    off = fx.Index(cols.valid(col).select(fx.Index(off), oob_off))
                dualwave._buffer_store_128(
                    dualwave._packed_o_128_vec(
                        traits, accs, dc, g, self.lane_div_32, self.elem_dtype
                    ),
                    off,
                    _o_store_reg_128=self.o_store_reg_128,
                    _store_atom_128=self.store_atom_128,
                    o_div=div,
                )


class BwdDkDvTileBody(dualwave.DualwaveKernelContext):
    """One streamed q tile: four GEMMs, one softmax, two barriers.

    A helper object rather than a nested `def`, because the loop this is called
    from uses the `range(..., init=[...])` / `yield` protocol, which only exists
    after the AST rewrite, so the body has to stay inside the traced function
    while the code it calls stays out of it.
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
        tight_registers,
        bias_fn=None,
    ):
        super().__init__(ctx)
        self.bias_fn = bias_log2e if bias_fn is None else bias_fn
        self.stream = stream
        self.reader = reader
        self.gemm = gemm
        self.softmax = softmax
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        self.tight_registers = tight_registers
        self.half_groups = ((0,), (1,)) if tight_registers else ((0, 1),)

    def _row_reader(self, base, scope, half, hdim):
        """A `ks -> pack` closure over one 32-row half of a staged tile.

        **The one line where `TIGHT_REGISTERS` changes what is emitted.** The
        tight arm reads each pack inside the MFMA that consumes it, so one is
        live at a time; the loose arm reads the whole half up front, which is
        `K_STEPS_QK * 4` VGPRs but lets every `ds_read_b128` issue before the
        first MFMA waits on it. Same `contract_d` either way -- it takes a
        callable precisely so this can be the only difference.
        """
        if const_expr(self.tight_registers):
            return lambda ks: self.reader.read_row_pack(base, scope, half, ks, hdim)
        packs = [
            self.reader.read_row_pack(base, scope, half, ks, hdim)
            for ks in range_constexpr(self.traits.K_STEPS_QK)
        ]
        return lambda ks: packs[ks]

    def load_neg_lse2(self, tile_base, h):
        """`-log2(e) * LSE` for the 16 rows half `h` holds. Overridden by the merged kernel (LSE -inf select)."""
        return self.softmax.load_row_values(
            self.ctx_ref.lse_rsrc, tile_base, h, self.ctx_ref.c_neg_log2e
        )

    def load_delta(self, tile_idx, tile_base, h, do_base, do_scope):
        """`Delta` for the 16 rows half `h` holds. Overridden by the merged kernel (rowsum(dO * O) in place)."""
        return self.softmax.load_row_values(
            self.ctx_ref.delta_rsrc, tile_base, h, self.ctx_ref.c_one_f
        )

    def run(self, tile_idx, buf_id, prefetch_idx, v_k, v_v, dv, dk):
        """Accumulate this tile into `(dv, dk)` and prefetch `prefetch_idx`.

        Barrier discipline, and both are needed:

        - the **top** barrier publishes this buffer's DMA, which every wave has
          just waited for;
        - the **bottom** one says every wave has finished reading it, which is
          what makes the prefetch below safe to overwrite it with.

        At two buffers the prefetch lands one tile ahead and overlaps the
        *next* tile's compute; at one it is issued immediately before the wait
        for it, which is the price head_dim 384 and 512 pay for their LDS. The
        body is the same either way.
        """
        traits = self.traits
        ctx = self.ctx_ref
        q_base = ctx.q_buf_base(buf_id)
        do_base = ctx.do_buf_base(buf_id)
        q_scope = dualwave._dualwave_lds_scope("k", buf_id)
        do_scope = dualwave._dualwave_lds_scope("v", buf_id)
        tile_base = tile_idx * fx.Index(traits.BLOCK_Q)

        # `vmcnt` retires in issue order, so leaving exactly the newer tiles'
        # groups outstanding retires this one and nothing else. The forward's
        # `NUM_DMA_K + NUM_DMA_V` idiom, over however many buffers are in play.
        _waitcnt_vm_n((traits.NUM_STREAM_BUFFERS - 1) * (ctx.NUM_DMA_K + ctx.NUM_DMA_V))
        _sched_barrier(0)
        _s_barrier()

        # -- GEMMs 1 and 2, then the softmax. `TIGHT_REGISTERS` decides
        #    whether the tile's two 32-row halves go through together or one
        #    at a time, the other half of the trade `_row_reader` makes.
        #
        # The halves are independent all the way to the packs. Together, the
        # peak holds two score accumulators, two dP, two LSE vectors, two
        # delta, two P and two dS -- about 96 f32 that a half-at-a-time order
        # never has live at once, since only the bf16 packs (8 VGPRs each)
        # survive a half. Apart, the scheduler has half as much independent
        # work to overlap the MFMA bursts with.
        #
        # `delta` is loaded *after* `probabilities` in both arms -- it is not
        # needed until `dscores`, and loading it earlier would hold its 16
        # registers across the LSE vector's.
        p_packs = [None] * 4
        ds_packs = [None] * 4
        for group in self.half_groups:
            s = {
                h: self.gemm.contract_d(
                    self._row_reader(q_base, q_scope, h, self.hdim_qk), v_k
                )
                for h in group
            }
            dp = {
                h: self.gemm.contract_d(
                    self._row_reader(do_base, do_scope, h, self.hdim_vo), v_v
                )
                for h in group
            }
            neg_lse2 = {h: self.load_neg_lse2(tile_base, h) for h in group}
            # The bias is read per accumulator element and folded into the
            # exponent, so it lands before `P` exists and therefore before
            # everything downstream. There is no runtime "does this tile need
            # one" guard: a bias build reads a bias for every live tile by
            # definition, and the descriptor bound is what makes a q row past
            # `seqlen_q` read zero rather than fault.
            bias2 = {h: None for h in group}
            if const_expr(traits.BIAS_TYPE):
                bias2 = {
                    h: self.bias_fn(ctx, self.softmax.q_rows(tile_idx, h))
                    for h in group
                }
            p_list = {
                h: self.softmax.probabilities(s[h], neg_lse2[h], bias2[h])
                for h in group
            }
            if const_expr(traits.CAUSAL):
                p_list = {
                    h: self.softmax.mask_if_clipped(p_list[h], tile_idx, h)
                    for h in group
                }
            delta = {
                h: self.load_delta(tile_idx, tile_base, h, do_base, do_scope)
                for h in group
            }
            # **After the row sum, before the accumulation.**
            # The softmax denominator is the *undropped* sum, and it is already
            # baked into the LSE the forward wrote, so dropout applies here and
            # nowhere earlier. It composes with the causal mask because that
            # one zeroes `P` first: a dropped survivor of a dead element is
            # still zero.
            keep = {h: None for h in group}
            if const_expr(traits.ENABLE_DROPOUT):
                keep = {
                    h: philox_keep(ctx, self.softmax.q_rows(tile_idx, h)) for h in group
                }
            ds_list = {
                h: self.softmax.dscores(p_list[h], dp[h], delta[h], keep[h])
                for h in group
            }
            if const_expr(traits.ENABLE_DROPOUT):
                # `dV` accumulates `keep * P`; the survivor scale is a constant
                # and rides the epilogue instead of 16 multiplies per tile.
                p_list = {
                    h: [
                        keep[h][r].select(fx.Float32(p_list[h][r]), ctx.c_zero_f)
                        for r in range_constexpr(16)
                    ]
                    for h in group
                }
            for h in group:
                p_packs[2 * h : 2 * h + 2] = self.softmax.pack_half(p_list[h])
                ds_packs[2 * h : 2 * h + 2] = self.softmax.pack_half(ds_list[h])

        # -- GEMM 3 and 4. Q is the reduction axis, so the transposed operand
        #    is read one output chunk at a time and dies with its MFMAs.
        for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
            dv[dc] = self.gemm.contract_q(
                self.reader.read_column_chunk(do_base, do_scope, dc), p_packs, dv[dc]
            )
        for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
            dk[dc] = self.gemm.contract_q(
                self.reader.read_column_chunk(q_base, q_scope, dc), ds_packs, dk[dc]
            )

        _s_waitcnt(traits.LGKMCNT_0_ONLY)
        _sched_barrier(0)
        _s_barrier()
        # Reading past the last tile is harmless and is what keeps this
        # branch-free: the descriptor bounds it, the zeros land in a buffer
        # nothing consumes, and the alternative is an `scf.if` around a DMA.
        self.stream.stage_q_tile(prefetch_idx, buf_id)
        self.stream.stage_do_tile(prefetch_idx, buf_id)
        return dv, dk


@dataclass(frozen=True)
class DkDvParts:
    """The classes `emit_dkdv_body` instantiates. The defaults are this kernel; the merged kernel substitutes subclasses."""

    ctx_cls: type = BwdDkDvKernelContext
    m32_tile: type = BwdDkDvTileBody
    m32_softmax: type = BwdDkDvSoftmaxHelper
    m16_tile: type = m16.M16TileBody
    m16_softmax: type = m16.M16SoftmaxHelper


DKDV_PARTS = DkDvParts()


def build_fmha_bwd_dkdv_gfx950_module_primary(meta, knobs, runtime_axes=()):
    """Build the dK/dV kernel for a resolved (meta, knobs) pair.

    `runtime_axes` ⊆ {"padded", "dropout"}: that axis becomes a launch-uniform branch on `rt_flags` (bit0 padded, bit1 dropout)
    that selects one of two whole bodies (both axes: four whole bodies, nested); geometry and schedule come from `knobs` unchanged.
    A runtime dropout axis needs a build whose traits enable dropout (it is the Philox state the dropout arm reads).
    """
    runtime_axes = frozenset(runtime_axes)
    assert runtime_axes <= {"padded", "dropout"}, runtime_axes
    if knobs.block_dmodel is None:
        raise ValueError(
            "knobs must be resolved: call `bwd_dkdv_knobs(arch, ...).resolve(meta)` first"
        )
    traits = knobs.build_traits(meta)
    assert (
        traits.ENABLE_DROPOUT or "dropout" not in runtime_axes
    ), "runtime_axes={'dropout'} needs a build with dropout enabled (meta.dropout=True): the Philox state is built only then"
    if traits.ENABLE_DROPOUT and philox.default_width("gfx950") != 32:
        raise ValueError(
            "the gfx950 dropout stream is Philox-4x32 (the forward's); philox.default_width('gfx950') must be 32"
        )
    BLOCK_DMODEL = knobs.block_dmodel
    PADDED_HEAD = knobs.padded_head
    # D columns at or below this are guaranteed real, so the streamed masks can
    # skip them. See `_masked_ks_steps`.
    HDIM_QK_FLOOR = knobs.hdim_qk_floor
    STRIDES_CONSTEXPR = knobs.strides_constexpr
    BUILD_SM_SCALE = meta.sm_scale
    NBUF = traits.NUM_STREAM_BUFFERS
    # Which MFMA family this build is. See `fmha_bwd_dkdv_m16_gfx950`.
    M16 = traits.MFMA_ROWS == 16
    # `(T, H)` logsumexp/delta. See `BwdDkDvInputMetadata.lse_layout_th`.
    # `lse_layout_th` does not reach the emitted code: the row read decodes
    # the layout at runtime and a varlen build carries both arms. It is in
    # the cache key so a caller who does pass it gets a distinct entry
    # rather than a silently shared one, and in the metadata so the
    # signature is stable; nothing else reads it.
    LSE_TH = bool(meta.lse_layout_th)
    # One knob for the whole register-against-ILP trade; see
    # `BwdDkDvTileBody._row_reader` and `_with_register_pressure`.
    TIGHT_REGISTERS = knobs.tight_registers
    MASKED_STEPS_ON = tuple(_masked_ks_steps(traits, HDIM_QK_FLOOR))
    # Perf knob (`BwdDkDvKnobs`): unset = the shipped code path.
    GQA_PACK = bool(knobs.gqa_pack)
    if GQA_PACK and not M16:
        raise ValueError("gqa_pack is written for the 16-row family")
    BIAS_FN = bias_log2e_packed if GQA_PACK else bias_log2e

    # Everything the build depends on, so that two builds differing anywhere
    # cannot collide in the JIT disk cache. `traits_cache_key` is every trait
    # field rather than `traits.cache_tag`'s subset; see there for what the
    # subset was silently getting wrong. It subsumes `NBUF` and `MFMA_ROWS`,
    # which are both `traits.` reads.
    # `TIGHT_REGISTERS` (a knob) and `LSE_TH` (from `meta`) reach the kernel
    # without passing through the traits, so they stay.
    _cache_tag = (
        traits_cache_key(traits),
        BLOCK_DMODEL,
        PADDED_HEAD,
        HDIM_QK_FLOOR,
        STRIDES_CONSTEXPR,
        BUILD_SM_SCALE,
        tuple(sorted(runtime_axes)),
        (TIGHT_REGISTERS, LSE_TH),
    ) + ((("gqa_pack", GQA_PACK),) if GQA_PACK else ())

    _lds_elem_dtype = dualwave.dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        # Q and dO tiles, interleaved per buffer: `[Q(0), dO(0), Q(1), dO(1)]`.
        stream: fx.Array[_lds_elem_dtype, traits.LDS_STREAM_TOTAL_SIZE, 16]

    def emit_dkdv_body(a, parts=DKDV_PARTS, ctx_kwargs=None):
        """The whole kernel body over one namespace `a` of the wire arguments; the kernel below is `a = args; emit`.

        `parts` names the context and tile-body classes (the merged kernel substitutes subclasses) and `ctx_kwargs`
        are extra context constructor arguments. Defaults reproduce this kernel exactly.
        """
        ctx_kwargs = {} if ctx_kwargs is None else ctx_kwargs
        Q = a.Q
        K = a.K
        V = a.V
        B = a.B
        DO = a.DO
        DK = a.DK
        DV = a.DV
        LSE = a.LSE
        Delta = a.Delta
        seqinfo_q0 = a.seqinfo_q0
        seqinfo_q1 = a.seqinfo_q1
        seqinfo_k0 = a.seqinfo_k0
        seqinfo_k1 = a.seqinfo_k1
        varlen_bits = a.varlen_bits
        num_seqlens = a.num_seqlens
        max_seqlen_q = a.max_seqlen_q
        max_seqlen_k = a.max_seqlen_k
        window_left = a.window_left
        window_right = a.window_right
        philox_seed_ptr = a.philox_seed_ptr
        philox_offset1 = a.philox_offset1
        philox_offset2 = a.philox_offset2
        idropout_p = a.idropout_p
        dropout_scale = a.dropout_scale
        num_head_q = a.num_head_q
        num_head_k = a.num_head_k
        hdim_qk = a.hdim_qk
        hdim_vo = a.hdim_vo
        sm_scale = a.sm_scale
        stride_q_batch = a.stride_q_batch
        stride_q_head = a.stride_q_head
        stride_q_seq = a.stride_q_seq
        stride_k_batch = a.stride_k_batch
        stride_k_head = a.stride_k_head
        stride_k_seq = a.stride_k_seq
        stride_v_batch = a.stride_v_batch
        stride_v_head = a.stride_v_head
        stride_v_seq = a.stride_v_seq
        stride_do_batch = a.stride_do_batch
        stride_do_head = a.stride_do_head
        stride_do_seq = a.stride_do_seq
        stride_dk_batch = a.stride_dk_batch
        stride_dk_head = a.stride_dk_head
        stride_dk_seq = a.stride_dk_seq
        stride_dv_batch = a.stride_dv_batch
        stride_dv_head = a.stride_dv_head
        stride_dv_seq = a.stride_dv_seq
        stride_b_batch = a.stride_b_batch
        stride_b_head = a.stride_b_head
        stride_b_seq_q = a.stride_b_seq_q
        rt_flags = a.rt_flags
        # Nine pointers in, nine nominal views out -- `wire_view` in
        # `fmha_dualwave_gfx950.py` has the argument. Everything below this line
        # is unchanged by the move, because every extent the kernel bounds a
        # descriptor with was already built from the strides and seqlens on the
        # wire rather than read off a tensor.
        Q = wire_view(Q)
        K = wire_view(K)
        V = wire_view(V)
        B = wire_view(B)
        DO = wire_view(DO)
        DK = wire_view(DK)
        DV = wire_view(DV)
        LSE = wire_view(LSE)
        Delta = wire_view(Delta)
        ctx = parts.ctx_cls(
            traits,
            strides=(
                stride_q_batch,
                stride_q_head,
                stride_q_seq,
                stride_k_batch,
                stride_k_head,
                stride_k_seq,
                stride_v_batch,
                stride_v_head,
                stride_v_seq,
                stride_do_batch,
                stride_do_head,
                stride_do_seq,
                stride_dk_batch,
                stride_dk_head,
                stride_dk_seq,
                stride_dv_batch,
                stride_dv_head,
                stride_dv_seq,
            ),
            sm_scale=sm_scale,
            num_head_q=num_head_q,
            num_head_k=num_head_k,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            padded_head=PADDED_HEAD,
            hdim_qk_floor=HDIM_QK_FLOOR,
            window_left=window_left,
            window_right=window_right,
            seqinfo=(seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1),
            varlen_bits=varlen_bits,
            num_seqlens=num_seqlens,
            Bias=B,
            bias_strides=(stride_b_batch, stride_b_head, stride_b_seq_q),
            philox=(philox_seed_ptr, philox_offset1, philox_offset2, None, None),
            idropout_p=idropout_p,
            dropout_scale=dropout_scale,
            Q=Q,
            K=K,
            V=V,
            DO=DO,
            DK=DK,
            DV=DV,
            Delta=Delta,
            LSE=LSE,
            seq_len=max_seqlen_q,
            seq_len_kv=max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
            block_table_stride=0,
            **ctx_kwargs,
        )
        ctx.GQA_PACK = GQA_PACK
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds()
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()
        ctx.init_kv_row()
        ctx.init_philox()
        # `-log2(e)` and `1.0`, so `load_row_values` is one function for both
        # row tensors: LSE has to cross into the base-2 domain the exponent
        # lives in, and delta is already in dP's units.
        ctx.c_neg_log2e = fx.Float32(-dualwave._LOG2E)
        ctx.c_one_f = fx.Float32(1.0)
        ctx.c_dropout_scale = fx.Float32(dropout_scale)

        _build_traits = traits

        def _arm_traits(drop_on):
            if bool(drop_on) == bool(_build_traits.ENABLE_DROPOUT):
                return _build_traits
            return _dc_replace(_build_traits, ENABLE_DROPOUT=bool(drop_on))

        def _make_arm(padded_on, drop_on):
            ctx.PADDED_HEAD = padded_on
            traits = _arm_traits(drop_on)
            ctx.traits = traits
            stream = BwdDkDvStreamLoader(ctx)
            if const_expr(M16):
                # The 16-row family. Same `run` signature, same barriers, same
                # prefetch discipline -- only the operand algebra differs, which is
                # why the loop below does not know which family it is driving.
                reader = m16.M16StreamReader(ctx)
                reader.init16(HDIM_QK_FLOOR)
                resident = m16.M16ResidentLoader(ctx)
                gemm = m16.M16GemmHelper(ctx)
                gemm.init16()
                store = m16.M16StoreHelper(ctx)

                def _m16_tile(lse_strided):
                    sm = parts.m16_softmax(ctx)
                    sm.LSE_STRIDED = lse_strided
                    return parts.m16_tile(
                        ctx,
                        stream=stream,
                        reader=reader,
                        gemm=gemm,
                        softmax=sm,
                        hdim_qk=hdim_qk,
                        hdim_vo=hdim_vo,
                        keep_fn=philox_keep,
                        bias_fn=BIAS_FN,
                    )

                tile_wide, tile_strided = _m16_tile(False), _m16_tile(True)
                n_acc = m16.d_chunks16(traits)
                zero_acc = Vec.filled(m16.ACC16, 0.0, fx.Float32).ir_value()
                acc_width = m16.ACC16
            else:
                reader = BwdDkDvStreamReader(ctx)
                reader.masked_steps = MASKED_STEPS_ON if padded_on else ()
                resident = BwdDkDvResidentLoader(ctx)
                gemm = BwdDkDvGemmHelper(ctx)
                store = BwdDkDvStoreHelper(ctx)

                def _m32_tile(lse_strided):
                    sm = parts.m32_softmax(ctx)
                    sm.LSE_STRIDED = lse_strided
                    return parts.m32_tile(
                        ctx,
                        stream=stream,
                        reader=reader,
                        gemm=gemm,
                        softmax=sm,
                        hdim_qk=hdim_qk,
                        hdim_vo=hdim_vo,
                        tight_registers=TIGHT_REGISTERS,
                        bias_fn=BIAS_FN,
                    )

                tile_wide, tile_strided = _m32_tile(False), _m32_tile(True)
                n_acc = traits.D_CHUNKS_PER_SHARD
                zero_acc = ctx.c_zero_v16f32
                acc_width = 16

            t0 = ctx.split_t0
            t_end = ctx.split_t_end
            scale_vec = Vec.from_elements([ctx.c_sm_scale], fx.Float32).broadcast_to(
                acc_width
            )
            drop_vec = Vec.from_elements(
                [ctx.c_dropout_scale], fx.Float32
            ).broadcast_to(acc_width)

            @flyc.jit
            def _dkdv_body(tile):
                """K/V resident, Q/dO streaming, two tiles per iteration.

                **The GQA group is the outer loop and the accumulators live across
                it**, which is the whole of the group reduction: several query heads
                reduce into this KV head, and their partial sums never leave f32
                registers. It costs no extra accumulator -- the fold lengthens the
                live range of the ones that were already carried, it does not add a
                level -- and it is why no scratch tensor, no LDS partials and no
                atomic appear anywhere here.

                At `num_kv_heads == num_heads` the group is one head wide, the
                outer loop runs once and emits no loop at all.
                """
                v_k = resident.load_resident("k", hdim_qk)
                v_v = resident.load_resident("v", hdim_vo)

                init_args = []
                for _ in range_constexpr(2 * n_acc):
                    init_args.append(zero_acc)
                group_results = init_args

                # **The trip count is the argument's, not the build's.** A bound read
                # from `traits.GQA_GROUP_SIZE` would be a single-iteration `scf.for`
                # the canonicaliser promotes away at MHA, which is right only when a
                # kernel is compiled per head-count pair and `_args` rejects a launch
                # whose runtime counts disagree with the build.
                #
                # AOTriton has neither half of that. `num_head_q` and `num_head_k`
                # are kernel arguments, not functional axes, so every compiled
                # kernel carries `GQA_GROUP_SIZE == 1`; and `_args` is a host-side
                # launch wrapper the C++ launcher never calls, so nothing catches
                # the disagreement. A build-time bound would fold the loop
                # away and each workgroup would sum only the *first* query head of
                # its group -- dK and dV short by the rest of the group, dQ untouched
                # (it is one program per query head and never folds).
                #
                # Runtime is what the rest of the parity path already assumes:
                # `ParityKernelContext.init_thread_mapping` derives `gqa_group` as
                # `num_head_q // num_head_k` for exactly this reason, and
                # `q_head_idx` -- the head this loop walks from -- is built from it
                # there. So the bound is *read* from `ctx.gqa_group` rather than
                # recomputed here: recomputing is what let a trait and a kernarg
                # drift apart, and a second spelling of the same quantity is the
                # same mistake one level down.
                #
                # Not `range_constexpr`: that would unroll the whole tile loop
                # `group` times, which is 8 copies of the largest region in the
                # kernel at MQA and would put the wide rungs through the build cap.
                group_trip = ctx.gqa_group
                if const_expr(GQA_PACK):
                    # `gqa_pack`: the group is already folded into the q row axis (see `m16.pack_split`), so there is one pass.
                    group_trip = fx.Index(1)
                for g, group_args in range(
                    fx.Index(0), group_trip, fx.Index(1), init=init_args
                ):
                    # Point the query side at this head. K, V, dK and dV do not
                    # move, and neither do the accumulators.
                    if const_expr(not GQA_PACK):
                        ctx.retarget_q_head(g)

                    # **Drain the pipeline before re-priming it.** The last tile of
                    # the previous head issued a prefetch into these same LDS
                    # buffers -- unconditionally, because making it conditional
                    # would put an `scf.if` around a DMA -- so the prologue below
                    # writes addresses that head's stale DMAs are still in flight
                    # to, and `vmcnt` is briefly counting two heads' issues at once.
                    #
                    # **Measured inert, and kept anyway.** A control build with
                    # this block removed is bit-identical on eight configurations,
                    # including `s=4096` at group 8 (deep pipeline) and the wide
                    # rungs where `NUM_STREAM_BUFFERS` collapses to 1 and the
                    # prefetch is issued immediately before the wait for it. The
                    # reason it is inert is that the staging is partitioned *by
                    # wave*, so no two waves write the same LDS bytes and a single
                    # wave's DMAs to one address retire in issue order -- the stale
                    # zeros always land before the fresh tile. That is an inference
                    # about hardware ordering rather than something read off a
                    # manual, and the drain is what makes the code not depend on
                    # it: 0.2-0.4% at group 8, once per head against a whole q
                    # walk. A wrong guess here costs a silently zeroed q tile.
                    #
                    # A build-time guard on `const_expr(traits.GQA_GROUP_SIZE > 1)`
                    # would skip it at group size 1, where there is no previous head
                    # at all -- and a guard is not cosmetic: left unconditional it
                    # measured **1.5% at head_dim 64**, where the kernel is
                    # shortest and a fixed prologue cost shows most. Every wider
                    # rung was within noise, which is exactly the shape of a
                    # constant added to the prologue.
                    #
                    # The guard has to follow the trip count it was reading: with
                    # the bound now runtime (see above) there is no build-time
                    # group size to fold against. So it becomes the *exact*
                    # condition rather than a conservative approximation of it --
                    # "is there a previous head", `g != 0`, which is true one
                    # iteration less often than the trait test would be and skips the
                    # drain on MHA the same way. `_stagger_extra_barrier_if_one`
                    # in `flash_attn_utils.py` is the same shape, an `scf.if` on a
                    # runtime scalar around `sched_barrier` + `s_barrier`.
                    #
                    # `g` is the `scf.for` induction variable, so the branch is
                    # workgroup-uniform and every wave reaches the same
                    # `s_barrier` -- which is the property a barrier under a branch
                    # needs and the reason this one is safe.
                    #
                    # **Inside a `@flyc.jit`, so the `if` is traced.** A bare
                    # Python `if` on an `fx` comparison out here is decided while
                    # tracing, not on the device, and what it decides is not the
                    # question being asked. The jit boundary is what turns it into
                    # the `scf.if` this comment describes.
                    @flyc.jit
                    def _drain_between_heads(g):
                        if g != fx.Index(0):
                            dualwave._waitcnt_vm_n(0)
                            dualwave._sched_barrier(0)
                            dualwave._s_barrier()

                    if const_expr(not GQA_PACK):
                        _drain_between_heads(g)

                    # Prime every buffer. From here each tile's DMA is issued by
                    # the body `NUM_STREAM_BUFFERS` tiles earlier, so this is the
                    # only staging whose latency is not covered by compute.
                    # `t0`, not a literal 0. Under causal the walk starts at the
                    # window's first live tile, and a prologue that assumed
                    # otherwise would stay right (a dead tile masks to nothing)
                    # while the tile cut goes inert, which only a timing
                    # assertion catches. The walk is the same
                    # for every head of the group: `t0`/`t_end` come from the KV
                    # block and the mask, neither of which the query head moves.
                    for b in range_constexpr(NBUF):
                        stream.stage_q_tile(t0 + fx.Index(b), b)
                        stream.stage_do_tile(t0 + fx.Index(b), b)

                    loop_results = group_args
                    for j, loop_args in range(t0, t_end, fx.Index(2), init=group_args):
                        dv = [loop_args[i] for i in range_constexpr(n_acc)]
                        dk = [loop_args[n_acc + i] for i in range_constexpr(n_acc)]
                        # Two tiles per iteration, one LDS buffer each when there
                        # are two. At one buffer both slots use it and the prefetch
                        # distance collapses; the arithmetic below is the same.
                        for slot in range_constexpr(2):
                            dv, dk = tile.run(
                                j + fx.Index(slot),
                                slot % NBUF,
                                j + fx.Index(slot + NBUF),
                                v_k,
                                v_v,
                                dv,
                                dk,
                            )
                        loop_results = yield dv + dk

                    group_results = yield [
                        loop_results[i] for i in range_constexpr(2 * n_acc)
                    ]

                dv = [group_results[i] for i in range_constexpr(n_acc)]
                dk = [group_results[n_acc + i] for i in range_constexpr(n_acc)]

                # `dS` is the gradient of the *scaled* logits, so `sm_scale` belongs
                # to `dK` and to nothing else. Once at the end rather than per tile:
                # a linear factor commutes with the accumulation.
                for dc in range_constexpr(n_acc):
                    dk[dc] = _fmul(Vec(dk[dc]), scale_vec, ctx.fm_fast)
                if const_expr(traits.ENABLE_DROPOUT):
                    # `dV = s * sum_i keep*P*dO`. A per-kernel constant, so it
                    # belongs on the accumulator once rather than on `P` every
                    # tile. `dK` does not get it: its `s` is already inside `dS`,
                    # where it multiplies only the `dP` term.
                    for dc in range_constexpr(n_acc):
                        dv[dc] = _fmul(Vec(dv[dc]), drop_vec, ctx.fm_fast)
                store.store_accs(dv, "dv", hdim_vo)
                store.store_accs(dk, "dk", hdim_qk)

            # A workgroup whose KV block is past *this* sequence's keys. Scalar
            # branch (the predicate is workgroup-uniform), so EXEC is untouched
            # across the transpose reads inside, which requires EXEC all 1s.
            # **One runtime branch, outside the tile loop, picking a whole body.**
            # `lse_row_addressing` decodes the logsumexp layout from `VarlenBits`
            # at runtime, so the kernel must serve either -- and putting the choice
            # inside `load_row_values` costs the wide arm its throughput, because
            # an `scf.if` there is a scheduling barrier the row loads cannot be
            # hoisted across (0.67x at head_dim 64, measured; the docstring on
            # `BwdDkDvSoftmaxHelper.load_row_values` has the table). Here each arm
            # is exactly the specialised code, at the price of
            # emitting the body twice.
            #
            # `pitch != 1` rather than the bits: one decode, no second reading, and
            # it is the sharper test -- at `num_head_q == 1` the two layouts
            # coincide and the wide arm is correct for both.
            # **A build without the varlen decode has no bits and cannot be TH**:
            # `varlen_bits` is 0, the layout is `_HT` and the pitch is 1, so there
            # is nothing to choose. Gating on the trait keeps such a build to one
            # body and confines the cost of the two arms to the builds that can
            # actually be handed the other layout.
            lse_strided = ctx.lse_pitch != fx.Index(1)
            active = ctx.active

            @flyc.jit
            def _run_body_varlen():
                if lse_strided:
                    _dkdv_body(tile_strided)
                else:
                    _dkdv_body(tile_wide)

            def _run_body():
                if const_expr(not traits.VARLEN or GQA_PACK):
                    _dkdv_body(tile_wide)
                else:
                    _run_body_varlen()

            return _run_body

        def _arm_flags(padded_on, drop_on):
            ctx.PADDED_HEAD = padded_on
            ctx.traits = _arm_traits(drop_on)

        _flags = fx.Int32(rt_flags)
        active = ctx.active
        if const_expr(not runtime_axes):
            _run_selected = _make_arm(PADDED_HEAD, traits.ENABLE_DROPOUT)
        else:
            if const_expr(len(runtime_axes) == 2):
                _f00, _f01 = (False, False), (False, True)
                _f10, _f11 = (True, False), (True, True)
                _cond_p = (_flags & fx.Int32(1)) != fx.Int32(0)
                _cond_d = (_flags & fx.Int32(2)) != fx.Int32(0)
                _a00 = _make_arm(*_f00)
                _a01 = _make_arm(*_f01)
                _a10 = _make_arm(*_f10)
                _a11 = _make_arm(*_f11)

                @flyc.jit
                def _run_selected():
                    if _cond_p:
                        if _cond_d:
                            _arm_flags(*_f11)
                            _a11()
                        else:
                            _arm_flags(*_f10)
                            _a10()
                    else:
                        if _cond_d:
                            _arm_flags(*_f01)
                            _a01()
                        else:
                            _arm_flags(*_f00)
                            _a00()

            else:
                if const_expr("padded" in runtime_axes):
                    _fa, _fb = (False, traits.ENABLE_DROPOUT), (
                        True,
                        traits.ENABLE_DROPOUT,
                    )
                    _cond = (_flags & fx.Int32(1)) != fx.Int32(0)
                else:
                    _fa, _fb = (PADDED_HEAD, False), (PADDED_HEAD, True)
                    _cond = (_flags & fx.Int32(2)) != fx.Int32(0)
                _run_a = _make_arm(*_fa)
                _run_b = _make_arm(*_fb)

                @flyc.jit
                def _run_selected():
                    if _cond:
                        _arm_flags(*_fb)
                        _run_b()
                    else:
                        _arm_flags(*_fa)
                        _run_a()

        if const_expr(active is None):
            _run_selected()
        else:

            @flyc.jit
            def _run_body_if_active():
                if active:
                    _run_selected()

            _run_body_if_active()

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def fmha_bwd_dkdv_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        rt_flags: fx.Int32,
    ):
        emit_dkdv_body(
            _WireArgs(
                Q=Q,
                K=K,
                V=V,
                B=B,
                DO=DO,
                DK=DK,
                DV=DV,
                LSE=LSE,
                Delta=Delta,
                seqinfo_q0=seqinfo_q0,
                seqinfo_q1=seqinfo_q1,
                seqinfo_k0=seqinfo_k0,
                seqinfo_k1=seqinfo_k1,
                varlen_bits=varlen_bits,
                num_seqlens=num_seqlens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                window_left=window_left,
                window_right=window_right,
                philox_seed_ptr=philox_seed_ptr,
                philox_offset1=philox_offset1,
                philox_offset2=philox_offset2,
                idropout_p=idropout_p,
                dropout_scale=dropout_scale,
                num_head_q=num_head_q,
                num_head_k=num_head_k,
                hdim_qk=hdim_qk,
                hdim_vo=hdim_vo,
                sm_scale=sm_scale,
                stride_q_batch=stride_q_batch,
                stride_q_head=stride_q_head,
                stride_q_seq=stride_q_seq,
                stride_k_batch=stride_k_batch,
                stride_k_head=stride_k_head,
                stride_k_seq=stride_k_seq,
                stride_v_batch=stride_v_batch,
                stride_v_head=stride_v_head,
                stride_v_seq=stride_v_seq,
                stride_do_batch=stride_do_batch,
                stride_do_head=stride_do_head,
                stride_do_seq=stride_do_seq,
                stride_dk_batch=stride_dk_batch,
                stride_dk_head=stride_dk_head,
                stride_dk_seq=stride_dk_seq,
                stride_dv_batch=stride_dv_batch,
                stride_dv_head=stride_dv_head,
                stride_dv_seq=stride_dv_seq,
                stride_b_batch=stride_b_batch,
                stride_b_head=stride_b_head,
                stride_b_seq_q=stride_b_seq_q,
                rt_flags=rt_flags,
            )
        )

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx950(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        rt_flags: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        num_kv_blocks = (
            fx.Index(max_seqlen_k) + fx.Index(traits.BLOCK_KV - 1)
        ) // fx.Index(traits.BLOCK_KV)
        # The grid's z extent counts **sequences**, which is `num_seqlens` when
        # a packed tensor holds several in one batch slot and `batch_size`
        # otherwise. A packed `(1, H, T, D)` call is `batch_size=1,
        # num_seqlens=N`; using the batch extent there launches one program for
        # N sequences.
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(traits.DAZ)
            else None
        )
        fmha_bwd_dkdv_gfx950_kernel(
            Q,
            K,
            V,
            Bias,
            DO,
            DK,
            DV,
            LSE,
            Delta,
            seqinfo_q0,
            seqinfo_q1,
            seqinfo_k0,
            seqinfo_k1,
            varlen_bits,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            window_left,
            window_right,
            philox_seed_ptr,
            philox_offset1,
            philox_offset2,
            idropout_p,
            dropout_scale,
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            sm_scale,
            stride_q_batch,
            stride_q_head,
            stride_q_seq,
            stride_k_batch,
            stride_k_head,
            stride_k_seq,
            stride_v_batch,
            stride_v_head,
            stride_v_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
            stride_dk_batch,
            stride_dk_head,
            stride_dk_seq,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            rt_flags,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            # **Head on the fast axis, and measured rather than inherited.**
            # Every workgroup here streams the whole of Q and dO, so putting
            # the KV block on the fast axis to make concurrent workgroups share
            # that slab is the obvious move -- and it is 12-15% *slower* at
            # every rung tried (512: 230 TF against 260; 384: 260 against 283;
            # 256: 390 against 433). The eight XCDs have separate L2s, so
            # sharing a slab duplicates it across all of them instead of
            # spreading distinct work. Same conclusion as the forward's, for a
            # different reason.
            #
            # **The head is the KV head.** Under GQA
            # the workgroup walks the group internally, so the grid shrinks by
            # the group size and each workgroup does that much more work --
            # total work unchanged, parallelism divided. At MHA the two counts
            # are equal and the grid is one workgroup per head and KV block.
            grid=(fx.Index(num_head_k), num_kv_blocks, bs_idx),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _resolve_window_args(window):
        """`(window_left, window_right)` for the wire, as signed i32.

        Always a pair, even for a build that ignores it, so every build shares
        one ABI and stays directly comparable -- the same reason the strides are
        passed under `strides_constexpr`. A *sentinel* rather than a bound goes
        on the wire for the fixed alignments: the kernel resolves it against
        each sequence's own lengths, which is the only correct thing to do once
        varlen means there is more than one pair to resolve against. The
        forward's `_resolve_window_args`, verbatim in intent.
        """
        if not traits.CAUSAL:
            if window is not None:
                # Dropping it silently returns dense attention's gradient:
                # right shape, finite, wrong. A window is only ever passed by a
                # caller who believes it is being applied.
                raise ValueError(
                    "window= requires a causal build; this one has causal=False"
                )
            return 0, 0
        if not traits.WINDOW:
            if window is not None:
                raise ValueError(
                    "this build is not compiled for windows; pass window=True in BwdDkDvInputMetadata"
                )
            return fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT
        if window is None:
            raise ValueError(
                "a window build requires window=(left, right); pass "
                "(fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT) for plain bottom-right causal"
            )
        wl, wr = window
        return int(wl), int(wr)

    def _check_8x_d_contract(named):
        """Refuse a layout the kernel's 8-wide accesses would overrun.

        **The input contract is 8xD, not 32xD.** Loads and stores are 8 columns
        wide, so a head_dim that is a multiple of 8 is a whole number of chunks
        and a plainly contiguous `(B, H, S, 24)` needs no padding of any kind.
        An odd head_dim still works, but only in an allocation with slack: the
        kernel touches `ceil8(hdim)` columns per row, so those extra elements
        must belong to the caller.

        **The grant covers the D axis and no other.** What the caller promises
        is that `t[b, h, s, :]` is `ceil8(hdim)` contiguous elements, for every
        `(b, h, s)` and not merely for the interior ones. What the caller does
        *not* promise is that anything else is contiguous: a guard page, an
        unmapped hole or another tensor may sit between two rows, between two
        heads or between two batches, since a stride says where the next row
        begins and never that the bytes in between are readable. So a buffer
        bound may round the D axis up -- it must, or the hardware's per-dword
        range check drops the tail chunk and takes the real column `hdim - 1`
        with it -- and may round no other axis by any amount. `_slab_span_elems`
        in `fmha_dualwave_gfx950.py` is where both halves are spent, and
        `flash_attn_func_gfx950`'s module docstring states the contract in full.

        Two separate requirements, and the pitch is only the first. *Alignment*:
        a row starts at `sum(index * stride)`, so every non-D stride must be a
        multiple of 8 for the 16-byte access to land aligned. *Slack*: the gap
        after a row is the smallest non-D stride, which for a BHSD tensor is
        the D pitch but for a BSHD one is `D` itself -- consecutive heads of the
        same token are adjacent, so a pitch check alone waves that through
        while the store corrupts the next head.
        """
        for name, t in named:
            d = t.shape[3]
            need = (d + 7) // 8 * 8
            if need == d:
                continue
            outer = [t.stride(i) for i in range(3) if t.shape[i] > 1]
            aligned = t.stride(3) == 1 and all(s % 8 == 0 for s in outer)
            slack = min(outer, default=need)
            if not aligned or slack < need:
                raise ValueError(
                    f"{name} has shape {tuple(t.shape)} strides {tuple(t.stride())}, which cannot hold "
                    f"a head_dim of {d}. {d} is not a multiple of 8, so the kernel reads and writes "
                    f"{need} columns per row and needs the D axis innermost, every other stride a "
                    f"multiple of 8, and {need - d} unused element(s) after each row. Allocate the last "
                    f"dimension as {need} and pass a [..., :{d}] view -- or use a head_dim that is a "
                    f"multiple of 8, which needs no padding at all."
                )

    def _args(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        LSE,
        Delta,
        batch_size,
        seqlen_q,
        seqlen_k=None,
        scale=None,
        window=None,
        varlen=None,
        num_seqlens=0,
        bias=None,
        dropout_p=None,
        philox_seed=None,
        philox_offset1=None,
        philox_offset2=0,
        stream=None,
    ):
        """Every kernel argument but the stream, in launch order.

        One place that turns tensors into the wire format, so `_launch` and
        `_compile` cannot drift apart -- the same shape, and the same reason,
        as the forward's `_args`.
        """
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        _ptrs, shape_meta, st = abi.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DK", DK), ("DV", DV)],
            q_heads=("DO",),
            k_heads=("DK", "DV"),
        )
        # Used for the checks and the strides, not the pointers: `prep_tensors`
        # builds those as `fx.Uint8`, which carries alignment 1. `wire_ptr`
        # types each operand from its own tensor instead -- see its docstring.
        del _ptrs
        num_head_q, num_head_k, hdim_qk, hdim_vo = shape_meta
        if num_head_q % num_head_k:
            raise ValueError(
                f"num_heads_q ({num_head_q}) must be a multiple of num_heads_k ({num_head_k}): each KV "
                "head owns a whole group of query heads and dK/dV sum over it."
            )
        if (num_head_q, num_head_k) != (traits.NUM_HEADS_Q, traits.NUM_HEADS_KV):
            # The group size sets the trip count of the outer loop and the grid
            # extent, so a build compiled for one shape cannot serve another.
            raise ValueError(
                f"this build is compiled for {traits.NUM_HEADS_Q} query heads over "
                f"{traits.NUM_HEADS_KV} KV heads; got {num_head_q} over {num_head_k}"
            )
        # dK follows Q's head dim and dV follows V's, which is what makes an
        # asymmetric build meaningful at all.
        if DK.shape[3] != hdim_qk or DO.shape[3] != hdim_vo or DV.shape[3] != hdim_vo:
            raise ValueError(
                f"dK must carry hdim_qk ({hdim_qk}) and dO/dV hdim_vo ({hdim_vo}); got "
                f"dK {DK.shape[3]}, dO {DO.shape[3]}, dV {DV.shape[3]}"
            )
        if hdim_qk > BLOCK_DMODEL or hdim_vo > BLOCK_DMODEL:
            raise ValueError(
                f"this build serves head dims up to {BLOCK_DMODEL}, got hdim_qk {hdim_qk} and "
                f"hdim_vo {hdim_vo}"
            )
        if not PADDED_HEAD and (hdim_qk != BLOCK_DMODEL or hdim_vo != BLOCK_DMODEL):
            raise ValueError(
                f"this build is not compiled for a padded head; it serves head_dim {BLOCK_DMODEL} "
                f"exactly, got hdim_qk {hdim_qk} and hdim_vo {hdim_vo}"
            )
        # Both extents share one mask floor, so both must sit above it.
        if HDIM_QK_FLOOR and min(hdim_qk, hdim_vo) <= HDIM_QK_FLOOR:
            raise ValueError(
                f"this build serves head dims in ({HDIM_QK_FLOOR}, {BLOCK_DMODEL}], got hdim_qk "
                f"{hdim_qk} and hdim_vo {hdim_vo}; build for the narrower rung, or pin "
                "hdim_qk_floor=0 to mask every column"
            )
        if PADDED_HEAD:
            _check_8x_d_contract(
                (("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DK", DK), ("DV", DV))
            )
        for name, t in (("DK", DK), ("DV", DV)):
            if t.dtype != Q.dtype:
                raise ValueError(
                    f"{name} must have Q's dtype ({Q.dtype}), got {t.dtype}"
                )
        # **The tensors must match what the build was compiled for.** Both
        # dtypes are two bytes, so every descriptor, stride and LDS offset is
        # identical between them and a mismatch changes nothing about the
        # addressing -- it is only the *bit interpretation* of the operands and
        # the MFMA opcode that differ. That is precisely why it has to be
        # checked here: handing f16 tensors to a bf16 build reads them as bf16,
        # which is finite, wrong by a factor near 2^112, and silent.
        # Compared by name rather than against a `torch` dtype object, because
        # this module does not import torch and should not start.
        want = {"bf16": "torch.bfloat16", "f16": "torch.float16"}[traits.DTYPE_STR]
        if str(Q.dtype) != want:
            raise ValueError(
                f"this build is compiled for dtype_str={traits.DTYPE_STR!r} ({want}); got {Q.dtype}. "
                "The two are the same width, so nothing downstream would notice."
            )
        # `row_tensor_arg` checks both the same way, which is what makes it safe
        # for the kernel to address them with one offset computation. Its return
        # is discarded: this kernel takes them as tensors, because it builds
        # buffer descriptors rather than dereferencing a raw pointer.
        abi.row_tensor_arg(LSE, "logsumexp", num_head_q, seqlen_q, varlen)
        abi.row_tensor_arg(Delta, "delta", num_head_q, seqlen_q, varlen)
        if Delta.shape != LSE.shape:
            raise ValueError(
                f"delta must have logsumexp's shape {tuple(LSE.shape)}, got {tuple(Delta.shape)}"
            )
        if varlen is None and LSE.shape[0] != int(batch_size) * num_head_q:
            raise ValueError(
                f"logsumexp must be ({int(batch_size) * num_head_q}, {seqlen_q}); got {tuple(LSE.shape)}"
            )
        # There is no build without the varlen decode: `make_traits` pins
        # `VARLEN=True` because the decode is unconditional, so a varlen
        # descriptor is never refused on that ground.
        # **There is no layout check.** The kernel decodes the logsumexp layout
        # from `VarlenBits` at runtime and serves either, so no call can disagree
        # with the build; a host-side check would in any case be no defence,
        # because AOTriton launches from C++ and never calls `_args`.
        # `lse_layout_th` is a build hint with no effect on correctness.
        # **There is no "requires a descriptor" check.** A caller who passes
        # nothing is the ordinary *dense* caller, and must be served.
        # They are, exactly: `varlen_args` gives `bits == 0`, and at zero bits
        # `decode_addressing` returns `(max_seqlen, 0, z)` -- dense addressing
        # -- with every array read behind a real branch, so the null `seqinfo`
        # pointers are never dereferenced. The cost of a dense call on a varlen
        # build is one not-taken scalar branch.
        # `abi.varlen_args` is gfx1201's, reused unedited: it encodes the same
        # wire format and it is where the two host-side checks live that no
        # kernel can make -- `batch_size` must be the tensor's batch extent
        # whatever the layout, and a packed `num_seqlens` must agree with the
        # length array.
        _vl = abi.varlen_args(
            STRIDES_CONSTEXPR, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens
        )
        # **The seed and the counter are the ones the *forward* reported**, not
        # the ones its caller passed: under graph capture the effective offset
        # is `*offset1 + offset2`, summed on the device. `abi.dropout_args` is
        # gfx1201's and takes the same (pointer, immediate) pair, so a caller
        # can hand the forward's `philox_seed_output` / `philox_offset_output`
        # straight back. The threshold and `1/(1-p)` are computed once here
        # rather than per element.
        *_dp, _dp_keepalive = abi.dropout_args(
            bool(traits.ENABLE_DROPOUT),
            dropout_p,
            philox_seed,
            philox_offset1,
            philox_offset2,
            device=Q.device,
            stream=stream,
        )
        if dropout_p is not None and not traits.ENABLE_DROPOUT:
            raise ValueError(
                "this build was not compiled for dropout; pass dropout=True in BwdDkDvInputMetadata"
            )

        # Checked both ways, because both silent failures return the right
        # shape: a bias build handed nothing would read whatever tensor filled
        # the slot, and a plain build handed a bias would return dense
        # attention's gradient to a caller who believes the bias was applied.
        if traits.BIAS_TYPE and bias is None:
            raise ValueError(
                "this build has bias=True and requires a (batch, num_heads, seqlen_q, seqlen_k) tensor"
            )
        if bias is not None and not traits.BIAS_TYPE:
            raise ValueError(
                "this build was not compiled for bias; pass bias=True in BwdDkDvInputMetadata"
            )
        if bias is not None:
            # **The bias follows Q on the query axis**, which is what the
            # addressing already does: the descriptor folds in `q_row_off` and
            # the head exactly as Q's slab does, so the shape rule is "match Q"
            # rather than a second layout to keep in step. Dense that reads as
            # `(b, h, seqlen_q, seqlen_k)`; under a packed varlen it reads as
            # `(1, h, total_q, max_seqlen_k)`, and under `varlen_padded` -- a
            # real batch axis -- it is the dense shape again. All three fall
            # out of one check, which is why it is written this way.
            #
            # The KV axis is *not* packed: it has no stride of its own (slot 3
            # is contiguous by contract), so each sequence's row spans
            # `seqlen_k` columns from 0 and the column index is the
            # within-sequence KV row the kernel already computes.
            if bias.shape[0] != Q.shape[0] or bias.shape[2] != Q.shape[2]:
                raise ValueError(
                    f"bias must share Q's batch and query-token extents; got {tuple(bias.shape)} "
                    f"against q {tuple(Q.shape)}. Under a packed varlen that means "
                    f"(1, num_heads, total_q, max_seqlen_k)."
                )
            if bias.shape[3] != seqlen_k:
                raise ValueError(
                    f"bias's last axis is the KV column and is not packed: it must be seqlen_k "
                    f"({seqlen_k}), got {bias.shape[3]}"
                )
            if bias.shape[1] != num_head_q:
                raise ValueError(
                    f"bias carries the *query* head count ({num_head_q}); got {bias.shape[1]}. The bias "
                    "shifts the scores, and there is one score matrix per q head even under GQA."
                )
            if bias.stride(3) != 1:
                raise ValueError(
                    f"bias needs a contiguous seqlen_k axis; strides are {tuple(bias.stride())}"
                )
            if bias.dtype != Q.dtype:
                raise ValueError(
                    f"bias must have Q's dtype ({Q.dtype}), got {bias.dtype}"
                )
        # The kernel's addressing is 32-bit per slab and per packed-row split;
        # past those limits a suppressed store lands on real data. AOTriton's
        # launcher does not run this function, so the same limits are the
        # caller's to honour there (see `check_launch_limits`).
        check_launch_limits(
            elem_bytes=traits.BF16_BYTES,
            block_q=traits.BLOCK_Q,
            gqa_pack=GQA_PACK,
            group=num_head_q // num_head_k,
            seqlen_q=_vl[
                5
            ],  # the longest sequences the kernel reads: max_seqlen_q / max_seqlen_k of a varlen call, seqlen_q / seqlen_k of a dense one
            seqlen_k=_vl[6],
            strides={
                n: (t.stride(1), t.stride(2))
                for n, t in (
                    ("Q", Q),
                    ("K", K),
                    ("V", V),
                    ("DO", DO),
                    ("DK", DK),
                    ("DV", DV),
                )
            },
            bias=(bias.stride(1), bias.stride(2)) if bias is not None else None,
        )
        # The sentinel is a real tensor whose contents are never read: every
        # use is behind `const_expr(traits.BIAS_TYPE)`, so a build without bias
        # emits no load at all. That is what makes `bias=None` bitwise
        # identical to a build compiled without it.
        bias_t = bias if bias is not None else DK
        bias_st = (
            tuple(int(x) for x in bias.stride()[:3]) if bias is not None else (0, 0, 0)
        )

        return (
            (
                # Nine pointers, not nine tensors. LSE and Delta are pinned to f32
                # rather than read off the tensor: both are required and both are
                # already checked f32 by `abi.row_tensor_arg`, so naming the type
                # here keeps the slot's type a property of the ABI rather than of
                # the call.
                wire_ptr(Q),
                wire_ptr(K),
                wire_ptr(V),
                wire_ptr(bias_t),
                wire_ptr(DO),
                wire_ptr(DK),
                wire_ptr(DV),
                wire_ptr(LSE, fx.Float32),
                wire_ptr(Delta, fx.Float32),
                int(batch_size),
                _vl[1],
                _vl[2],
                _vl[3],
                _vl[4],
                _vl[0],
                int(num_seqlens),
                *_vl[5:],  # max_seqlen_q, max_seqlen_k -- the decode's MAX fallback
                *_resolve_window_args(window),
                _dp[0],
                _dp[1],
                _dp[2],
                _dp[3],
                _dp[4],
                num_head_q,
                num_head_k,
                hdim_qk,
                hdim_vo,
                abi.resolve_scale(
                    Q,
                    scale if scale is not None else BUILD_SM_SCALE,
                    PADDED_HEAD,
                    1.0 / (BLOCK_DMODEL**0.5),
                ),
                *st,
                *bias_st,
                int(hdim_qk != BLOCK_DMODEL or hdim_vo != BLOCK_DMODEL)
                | (2 if dropout_p is not None else 0),
            ),
            stream,
            _dp_keepalive,
        )

    def _launch(*args, **kwargs):
        # `keepalive` holds the device tensors behind an int seed / offset: only their raw pointers reach the kernel,
        # so they must stay referenced until the launch is enqueued.
        packed, stream, keepalive = _args(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            result = abi.run_compiled(
                _COMPILED,
                launch_fmha_bwd_dkdv_gfx950,
                *packed,
                stream if stream is not None else fx.Stream(None),
            )
        del keepalive
        return result

    def _compile(*args, **kwargs):
        packed, stream, _keepalive = _args(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return flyc.compile(launch_fmha_bwd_dkdv_gfx950, *packed, fx.Stream(stream))

    _launch.compile = _compile
    _launch.traits = traits
    _launch.knobs = knobs
    _launch.emit = emit_dkdv_body
    _launch.SharedStorage = SharedStorage
    _launch.cache_tag = _cache_tag
    return _launch


def build_fmha_bwd_dkdv_gfx950_module(arch="gfx950", **kwargs):
    """Keyword front end: name a problem, get the policy's schedule."""
    from dataclasses import fields as _fields

    meta_fields = {f.name for f in _fields(BwdDkDvInputMetadata)}
    meta = BwdDkDvInputMetadata(**{k: v for k, v in kwargs.items() if k in meta_fields})
    knob_kwargs = {k: v for k, v in kwargs.items() if k not in meta_fields}
    return build_fmha_bwd_dkdv_gfx950_module_primary(
        meta, bwd_dkdv_knobs(arch, **knob_kwargs).resolve(meta)
    )
