# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Flash-attention **backward dQ (and dB)** for gfx950 -- AOTriton's `bwd_kernel_dq`.

B2 and B3 of `sdpa-bwd-plan-gfx950.md`. Dense, non-causal, bf16, the full
`LADDER` from head_dim 32 to 512, with the 8xD input contract, padded heads and
asymmetric `hdim_qk`/`hdim_vo`. The dK/dV half is B1 and lives in its own file;
there is deliberately no fused kernel (plan section 4).

    P  = exp2(qk_scale * S - lse2)        S  = Q.K^T,  qk_scale = sm_scale*log2e
    dP = dO . V^T
    dS = P * (dP - delta)                 delta = rowsum(dO*O), a host argument
    dQ = sum_j dS . K * sm_scale
    dB = dS                               (bias-gradient builds only)

--- The whole design, in one observation --------------------------------------

**Every one of the three GEMMs is a GEMM the forward already emits**, once the
right tensor is staged in the right LDS layout. Nothing below derives a new
lane map, and that is the point: the forward's `V` read path is validated end
to end, and plan section 3 says to reuse it rather than re-derive it.

| this kernel | forward equivalent | A operand (M axis) | B operand (N axis) |
|---|---|---|---|
| `S  = Q.K^T`  | `qk` | K, `load_k` | Q, `ParityQLoader` |
| `dP = dO.V^T` | `qk` | V, `load_k` | dO, same loader |
| `dQ = dS.K`   | `pv` | K^T, `load_v` | dS, packed like P |

The score-shaped accumulators are all `[kv][q]` -- M is the KV token, N is the
query row -- which is exactly the forward's `v_s`, so `seq_pad_mask_if_needed`,
`sub_m`, `exp2` and `cast_p` apply unchanged. `dS` reaches the third GEMM
through `cast_p`, so it carries the *same* K permutation `P` does, which is why
`load_v`'s transpose read lands K in the operand registers the MFMA wants with
no further shuffle.

--- Two LDS tiles, and the read that makes it two ------------------------------

GEMM1 wants K with the KV token on the MFMA's M axis and `d` contracted; GEMM3
wants K with `d` on M and the KV token contracted. Those are the forward's K
path and its V path, and the two stagings differ **only in the line padding**
(`SMEM_K_PAD` 8 elements against `SMEM_V_PAD` 32). B2 answered that by staging
K twice. B3 cannot: three slots is 199 KB at head_dim 512 against a 163840 B
cap, so the double staging is what stops the ladder rather than merely costing
DMA.

So the K tile is staged **once, in the V layout**, and read two ways:

    (V, buf 0) <- K   -> load_v(0)          GEMM3, the stock transpose read
                      -> load_k_packs()     GEMM1, the K path on V-pitch lines
    (K, buf 0) <- V   -> load_k_packs()     GEMM2, the stock K path
    buf 1                unused, and not allocated

`LDS_KV_TOTAL_SIZE` drops to one buffer -- `SMEM_K_TILE_ELEMS +
SMEM_V_TILE_ELEMS`, 133 KB at head_dim 512 -- which is a single trait field
(`BwdDqTraits`, `make_bwd_dq_traits`).

**The K tile is the one in the V region, not the other way round, and that is
deliberate.** One of the two readers has to be re-pointed at the other pitch,
and the K path is a plain `llvm.LoadOp` whose whole addressing is two
expressions; the transpose path is `ds_read_b64_tr_b16` with alias scopes, an
even-VGPR-pair constraint and two open hazards against it (`sdpa_lore_gfx950`).
Leave the fragile one stock. `BwdDqKvLdsToVgprLoader` re-points the other by
scoping two trait fields, so the *formula* is still the shared one.

The alias scopes stay truthful: the K tile is read under `lds_v0` by both of
its readers and the V tile under `lds_k0`, so no scope claims two regions are
disjoint when they are the same memory.

--- Conventions this kernel must not get wrong --------------------------------

Three, and each is a *silent* wrong answer if broken, which is why the tests
check them against our own forward rather than only against torch:

- **`qk_scale = sm_scale * log2e` is applied to the f32 scores, `sm_scale`
  alone to the dQ accumulator at the end.** AOTriton spells them
  `p = exp2(qk_scale*qk - l_i)` and `dq *= sm_scale`. The forward used to fold
  `qk_scale` into Q and round the product back to bf16 while this kernel
  deliberately did not; `BwdDqSoftmaxHelper.scale_and_sub_lse` has the
  measurement that says why, and the forward has since been moved onto the
  same arithmetic (`ParityGemmHelper.scale_scores`). The two now agree.
- **Neither Q nor `dO` is pre-scaled.** `ParityQLoader.scale_all` now raises
  rather than scaling, so the Q half of this is enforced; `dO` is not, and
  pre-scaling it multiplies every gradient by `qk_scale` with nothing checking
  shapes.
- **LSE is read in natural units and converted here.** The forward writes
  `m_row*ln2 + ln(l)`, so `lse2 = lse * log2e` (AOTriton's `l_i = ... *
  RCP_LN2`). `fmha.lse_row_addressing` owns the layout for both LSE and delta.

--- The two head extents are crossed, and that is not a detail ----------------

In the forward, the tensor read through the K register path has the *qk*
extent and the tensor written through the O store has the *vo* one. **Both are
the other way round here**, because GEMM2 reads V through the K path and the
store writes dQ, which is Q-shaped:

- two `BwdDqKvLdsToVgprLoader` instances, one per tile, differing in the LDS
  pitch *and* in which extent their padded-head mask is written against
  (plus `HDIM_VO_FLOOR`, the vo counterpart of `HDIM_QK_FLOOR`);
- `BwdDqStoreHelper`, which points the inherited store at dQ's descriptor,
  stride, sentinel and `hdim_qk` width.

They coincide in every symmetric build, so only `test_asymmetric_hdim` can tell
the fix from its absence.

--- B3.5: what a 16-row family still needs, after the shape is named -----------

`BwdDqTraits` now names `MFMA_M/N/K` and derives `ACC_ELEMS`,
`SCORE_MSTEPS` and `OPERAND_LANE_ELEMS` from them, and every literal in this
file that meant one of those is now written as one. That change is a no-op
today -- the ISA is byte-identical, same 260 VGPRs at head_dim 128 -- so what
is left is exactly the list below, and nothing else.

**Two findings that do not need the lane-map probe**, both in
`fmha_tuning_bwd_dq_gfx950`:

- `v_mfma_f32_16x16x16_bf16` is **4 passes for 8192 flops** against
  `32x32x16`'s **8 for 32768** (`SISchedule.td`, `SIDPGFX950FullSpeedModel`).
  It is *half rate*, so a family built on it cannot beat the 32-row one
  anywhere the 32-row one fits. `16x16x32` is 4 passes for 16384 -- full rate,
  16 rows.
- `register_demand` says 16 rows takes head_dim 384 from 144 registers over the
  file to **fitting**, and 512 from 336 over to **40 over**. So 16 rows
  finishes 384 and leaves 512 needing one more lever.

**16x16x32 is also structurally the cheaper port**, and that is a third
argument for it: `OPERAND_LANE_ELEMS` stays 8, so `_pack_p_v8_slices`,
`_bf16_trunc_pack_v8` and the v8 shape of every LDS read are unchanged. At
16x16x16 all of them become 4-wide.

What still has to be re-derived, in dependency order:

1. **`SCORE_MSTEPS == 2` is structural, not parametric.** The softmax path is
   written as an explicit `(s_lo, s_hi)` *pair* throughout
   `flash_attn_utils` -- `_score_pair_to_lists`, `_reduce_score_pair`,
   `_sub_score_pair`, `_exp2_score_slice`, `_pack_p_v8_slices`,
   `seq_pad_mask_inplace`. At 16 rows and `BLOCK_N` 64 there are four M steps.
   **`BLOCK_N` 32 keeps the pair, and it is also what closes head_dim 512.**
   `register_demand` at `(16 rows, BLOCK_N 32)` is 404 at head_dim 512 and 308
   at 384, against 552 and 424 at `BLOCK_N` 64 -- so the cheapest port and the
   register fix are the same choice, and there is no need to generalise the
   pair. Settle this before anything else.
2. `_seq_pad_score_threshold` -- the element -> KV column map, derived from the
   32x32 accumulator layout. The KV tail mask and the dB store both read it,
   so they stay consistent for free once it is right. `fmha_mfma16_gfx950`
   gives the replacement: a lane's four accumulator elements are four
   *contiguous* KV tokens at `4 * (lane // 16)`, where at 32 rows they are the
   scattered `8*(i//4) + 4*(lane//32) + (i%4)`.
3. The `permlane32_swap` O store (`_fused_o_128_dwords`): it transposes the
   `[d][q]` accumulator for a 128-bit row-major store, and `permlane32_swap` is
   a 32-lane primitive.
4. `_init_dualwave_q_row`'s `q_row = ... + lane_mod_32`, which the LSE/delta
   row load reads. At 16 rows it is `lane % 16`; the two row inputs stay
   per-lane scalars, because the query row is the accumulator's *n* axis here
   (it is dK/dV, where the query row is *m*, that gets the `dwordx4`).
5. The transpose read for GEMM3. **Answered by the probe, and favourably**
   (`fmha_mfma16_gfx950`): it serves both 16-row shapes from *one* staged tile,
   so the four-LDS-tile risk does not materialise. Better than that for this
   kernel -- **the 16-row maps carry no permutation on the contraction index**,
   where the 32-row one does. GEMM3 currently works because `cast_p`'s pack
   order happens to reproduce the 32-row read's `k` permutation; at 16 rows
   `dS` only has to be in plain order, so that coincidence stops being
   load-bearing.

Items 2-4 live in `flash_attn_utils.py`, which is imported and never edited, so
each becomes an override here.

--- f16 and bf16, and where the operand dtype actually reaches ----------------

Both are supported; `BwdDqKnobs.build_traits` refuses anything else by name,
matching gfx1201's `assert dtype_str in ("f16", "bf16")`.

**The operand dtype reaches only two places**, and knowing that is what makes
f16 cheap here: the `dS` pack that feeds GEMM3's B operand
(`_bf16_trunc_pack_v8`, which already branches on `DTYPE_STR`) and the dQ
store. Everything with an exponent in it -- the score, `lse2`, `delta`, the
softmax, `dP`, the dropout survivor scale, the bias add -- is **f32 in both
builds**. So f16's narrower range (65504 against bf16's fp32 range) is exposed
at those two narrowings and nowhere else.

Two consequences worth stating rather than rediscovering:

- The `lse2` floor and the dropout ordering were both derived under bf16, and
  both carry, for the same reason: they are about f32 intermediates. See
  `BwdDqKernelContext.load_row_scalars`.
- **The dropout survivor scale is applied to the f32 `dP`, not folded into
  `dO`.** B6 chose that on precision grounds -- folding would round `dO`
  through the operand type a second time -- and it is also the f16-safe
  choice, because `1/(1-p)` on an f16 operand is a multiply into a 65504
  ceiling where on f32 it is not.

What is *not* guarded, deliberately and in line with AOTriton and gfx1201: a
`dS` or `dQ` magnitude beyond 65504 saturates in an f16 build where a bf16
build would carry it. That needs inputs far outside anything these tests or a
trained model produce, and clamping it would cost a per-element `min` on the
hot path to change one implausible case into a different wrong answer.

--- Not implemented, deliberately ---------------------------------------------

Causal, windows, varlen, dropout, bias *input*, split-K, paged, `D_STAGES`,
d-axis sharding. `BwdDqKnobs.build_traits` refuses each by name rather than
ignoring it -- every one of them would otherwise build, run and return a
correctly-shaped wrong answer, and `D_STAGES` nearly did: the forward's knob
policy turns it on above head_dim 256 and the inherited GEMM helpers then
reduce over one stage of the head dim while this loop never advances the stage.

--- Tensor argument order is the ABI ------------------------------------------

    q, k, v, b, do, dq, db, lse, delta

Four groups, and the grouping is the mnemonic: the **forward's inputs**
(`q, k, v, b`), then the **backward's input** (`do`), then this kernel's
**outputs**, then the **lower-rank** tensors (`lse`, `delta`, both rank 2).

It is also AOTriton's order -- `bwd_kernel_dq(Q, K, V, B, sm_scale, DO, DQ, DB,
L, D, ...)` and `bwd_kernel_dk_dv(Q, K, V, B, sm_scale, DO, DK, DV, L, D, ...)`
-- so a reader moving between the Triton reference and this file does not have
to re-derive the mapping, and neither does anyone eventually dispatching the
compiled hsaco directly.

`b` sits in the forward-input group even though no build here reads it yet: the
slot is held so adding bias in B6 does not move the wire format. It was
initially placed after `delta`, at the end, which is where an unused argument
naturally lands and is exactly why the grouping is written down rather than
left to accrete.
"""

from dataclasses import replace
from types import SimpleNamespace as _WireArgs

from kernels.attention_bwd_cdna import fmha_abi_gfx1201 as abi
from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha
from kernels.attention_bwd_cdna.fmha_bwd_dq_m16_gfx950 import make_m16_dq_body
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import _fmax, _fmul, _fsub
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvGmemToLdsLoader,
    ParityKvLdsToVgprLoader,
    ParityQLoader,
    ParitySoftmaxHelper,
    traits_cache_key,
    ParityStoreHelper,
    _bias_slab_num_records_bytes,
    _fadd,
    _score_column_runs,
    _slab_span_elems,
    exp2_wait_state,
    mfma_operand_wait_state,
    wire_ptr,
    wire_view,
)

# Private to `fmha_dualwave_gfx950`, and imported anyway: they are the
# granule-general spellings of the K read's per-lane base and k-step offset,
# and the alternative is a second copy of two formulas the read and write sides
# of an LDS layout both depend on. Reaching across within one kernel family
# beats transcribing.
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    _k_read_base as _parity_k_read_base,
)
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    _ks_offset as _parity_ks_offset,
)
from kernels.attention_bwd_cdna.fmha_tuning_bwd_dq_gfx950 import (
    BwdDqKnobs,
    bwd_dq_knobs,
)
from kernels.attention_bwd_cdna.fmha_tuning_gfx950 import FmhaInputMetadata
from kernels.attention_bwd_cdna.gfx950_standalone import buffer_ops, dualwave
from kernels.attention_bwd_cdna.philox import Philox

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

KERNEL_NAME = "fmha_bwd_dq_gfx950_kernel"

__all__ = [
    "KERNEL_NAME",
    "BwdDbStoreHelper",
    "BwdDqKernelContext",
    "BwdDqKvGmemToLdsLoader",
    "BwdDqKvLdsToVgprLoader",
    "BwdDqSoftmaxHelper",
    "BwdDqStoreHelper",
    "BwdRowInputLoader",
    "BwdSecondaryQLoader",
    "build_fmha_bwd_dq_gfx950_module",
    "build_fmha_bwd_dq_gfx950_module_primary",
]

_s_barrier = dualwave._s_barrier
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier


def bias_chunk_contract_ok(data_ptr, strides, seqlen_k):
    """The caller contract of an LDS-staged bias (`BwdDqKnobs.bias_stage`): the pointer 16-byte aligned, the batch/head/row strides
    (`strides`, in elements) multiples of 8, and a non-zero row stride at least `ceil8(seqlen_k)`, so the last row's final 16-byte chunk
    lies inside the allocation. What PyTorch's padded attn_bias already is."""
    return not (
        data_ptr % 16
        or any(int(x) % 8 for x in strides[:3])
        or (int(strides[2]) and int(strides[2]) < -(-int(seqlen_k) // 8) * 8)
    )


def _carried(values, count):
    """An `scf.for`'s carried values as a list of `count`, however many it hands back.

    **A loop with exactly one carried value hands it back unwrapped**, because
    an `scf.for` with one result is a value rather than a tuple. This kernel
    carries `D_CHUNKS` accumulators and nothing else, so head_dim 32 -- the one
    rung where `D_CHUNKS == 1` -- is the only place that bites, and it bites
    two frames away: `loop_results[0]` on a `vector<16xf32>` returns the *first
    f32 element*, and the failure surfaces inside `_scale_o_accs` as "Cannot
    cast type to VectorType". The forward never sees this because it always
    carries `m_row` and `l_row` alongside.
    """
    if const_expr(isinstance(values, (list, tuple))):
        return list(values)
    if const_expr(count != 1):
        raise AssertionError(
            f"expected {count} carried values, got one unwrapped value"
        )
    return [values]


_COMPILED = {}

# The forward's, unchanged: this kernel is the same instruction mix on the same
# schedule primitives, so a different set here would only make the two
# incomparable at the ISA level.
_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}


class BwdDqKernelContext(ParityKernelContext):
    """The forward's parity context plus dQ, dO, dB and the two row inputs.

    Subclassed rather than ported, per the contract: the strides, the padded
    head, the varlen decode and the descriptor machinery are all inherited.
    What is added is this kernel's output (`dQ`), one more Q-shaped input
    (`dO`), one score-shaped output (`dB`), and the LSE/delta pair.

    **There is no `O` here.** dQ used to be handed to the base class's `O`
    slot, which made `stride_o_*` dQ's strides, `o_div` dQ's descriptor and
    `hdim_vo` -- the head dim of V or O -- the bound on a tensor that is
    Q-shaped and therefore `hdim_qk` wide. At an asymmetric head dim that
    bound is simply the wrong number, and the D-tail stores it was supposed to
    admit were dropped instead. dQ is spelled `DQ` here, with `stride_dq_*`,
    `dq_div` and `dq_oob_off`; the base's `O` stays None and its O view is not
    built. `BwdDqStoreHelper` is the one place upstream's `o_*` store
    vocabulary is bound to them.
    """

    def __init__(
        self,
        traits,
        *,
        dq_strides,
        do_strides,
        db_strides=(0, 0, 0),
        DQ=None,
        DO=None,
        DB=None,
        Delta=None,
        **kwargs,
    ):
        super().__init__(traits, **kwargs)
        self.stride_dq_batch, self.stride_dq_head, self.stride_dq_seq = dq_strides
        self.stride_do_batch, self.stride_do_head, self.stride_do_seq = do_strides
        # `_seq_q` for the reason `ParityKernelContext` gives for the bias
        # input's: dB is `(batch, head, seqlen_q, seqlen_k)` and a bare `_seq`
        # does not say which of the two it is.
        self.stride_db_batch, self.stride_db_head, self.stride_db_seq_q = db_strides
        self.DQ = DQ
        self.DO = DO
        self.DB = DB
        self.Delta = Delta

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_dq_seq_v = fx.Index(self.stride_dq_seq)
        self.stride_do_seq_v = fx.Index(self.stride_do_seq)

    def init_descriptors(self, **kwargs):
        """The forward's four views, plus dQ's, dO's and dB's.

        `dO` gets exactly Q's treatment -- same slab shape, same row origin,
        same `seqlen_q` bound -- because it *is* Q-shaped: (batch, head, q row,
        d), with the vo head dim rather than the qk one. The bound is what
        makes a row past `seqlen_q` read zero instead of faulting, which is the
        whole reason the ragged tail needs no branch.

        Each of the three carries the width of *its own* rows, which is the
        point of `_slab_span_elems`: `stride_seq` is the distance between rows
        and is only the width of one when the tensor is contiguous in
        `(.., seqlen, hdim)` order. dQ and dO differ in that width -- dQ is
        Q-shaped and `hdim_qk` wide, dO is O-shaped and `hdim_vo` wide -- and
        an asymmetric build is where the difference shows.
        """
        traits = self.traits
        super().init_descriptors(**kwargs)
        # dQ is Q-shaped -- (batch, head, q row, d) with the *qk* head dim --
        # so it gets Q's slab treatment against `hdim_qk`. Not `hdim_vo`: at
        # an asymmetric head dim the two are different numbers and only one of
        # them is dQ's row width.
        self.dq_div, self.dq_oob_off = self.q_row_slab(
            self.DQ,
            self.stride_dq_batch,
            self.stride_dq_head,
            self.stride_dq_seq,
            self.hdim_qk,
        )
        self.do_div = self._slab_view(
            self.DO,
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
            self.hdim_vo,
        )
        self.do_gmem_elem_offset = self.q_start * self.stride_do_seq_v
        if const_expr(traits.STORE_DB):
            # **Whether dB is wanted is a runtime question, not a build one.**
            # `STORE_DB` above only decides whether the store *exists*; a build
            # that has it still has to serve a caller who passes a bias but does
            # not want its gradient, which AOTriton spells as an all-zero dB
            # stride triple over a null DB (`bwd_kernel_dq.py`'s `store_db`).
            # PyTorch takes that path on every bool-masked SDPA: the mask becomes
            # an additive bias that is not a leaf, so attention_backward.cu hands
            # over `empty_t4` for DB.
            #
            # Not currently a fault here, unlike gfx1201's flat store: the buffer
            # bound below collapses to 0 records when `stride_db_seq_q` is 0, so
            # the hardware drops every store and the null base is never touched.
            # That is an accident of the `num_records` arithmetic rather than
            # anything this kernel says, and it does not cover a null DB carried
            # with a non-zero row stride. Say it instead.
            self.db_store_enabled = (
                (fx.Int64(self.stride_db_batch) != fx.Int64(0))
                | (fx.Int64(self.stride_db_head) != fx.Int64(0))
                | (fx.Int64(self.stride_db_seq_q) != fx.Int64(0))
            )
            # Same slab shape as the forward's bias descriptor -- dB is indexed
            # (batch, head, q row, kv col) and the KV axis is contractually
            # contiguous -- and a raw resource for the same reason: the stores
            # are per-lane at an address the lane computes, which is
            # `buffer_ops.buffer_store`'s shape and not the copy atom's.
            # First element past the *untightened* span. A store redirected
            # here is dropped by the hardware bound; see `BwdDbStoreHelper`.
            # It stays untightened deliberately: a sentinel only has to sit at
            # or past `num_records`, and the bound below only shrinks, which
            # moves it further out of range rather than back into it.
            self.db_oob_off = self.seqlen_q_v * fx.Index(self.stride_db_seq_q)
            # Bounded through the forward's own helper, and for its reason: a
            # BSHD caller's `stride_db_seq_q` is `num_heads * seqlen_k`, so
            # `seqlen_q * stride_db_seq_q` overshoots this head's slab by the
            # other heads of the last row -- which for the last (batch, head)
            # is off the end of dB. Storing there would corrupt a neighbouring
            # head's gradient, or fault.
            #
            # The load-side caveat in that helper's docstring -- a wide read
            # losing the last column to the per-dword range check at an odd
            # `seqlen_k` -- has no store-side counterpart here: this is a
            # *2-byte* store per element, already predicated on
            # `col < seqlen_k`, so every live one lands inside the tight bound
            # whatever the parity.
            #
            # Clamped to the sentinel: a redirected store is dropped only while
            # `num_records <= db_oob_off`. The helper bounds a zero row stride at
            # one row (`seqlen_k`, a row-broadcast bias is a valid load slab), but
            # for dB an all-zero stride triple means "not requested" and puts the
            # sentinel at 0, so the store bound must be 0 there too.
            self.db_rsrc = buffer_ops.create_buffer_resource(
                self.DB,
                max_size=False,
                num_records_bytes=arith.minui(
                    _bias_slab_num_records_bytes(
                        self.seqlen_q_v,
                        self.seqlen_kv_v,
                        self.stride_db_seq_q,
                        traits.BF16_BYTES,
                    ),
                    as_mlir_value(self.db_oob_off * fx.Index(traits.BF16_BYTES)),
                ),
                base_byte_offset=as_mlir_value(
                    self._slab_byte_base(
                        self.stride_db_batch,
                        self.stride_db_head,
                        self.stride_db_seq_q,
                        self.q_row_off,
                        self.q_head_idx,
                    )
                ),
            )
        if const_expr(traits.BIAS_STAGE == "lds"):
            self.init_bias_descriptors()

    def init_bias_descriptors(self):
        """The bias view of an LDS-staged build, bounded at the row's 8-element ceiling.

        The incumbent's `bias_rsrc` ends exactly on the last valid element, and
        gfx950 range-checks a multi-dword buffer op **per dword**, so a 16-byte
        chunk or `dwordx2` run holding the last column of an odd `seqlen_k`
        loses that column (`_add_bias_inplace` documents it and narrows to
        2-byte loads). An LDS-staged build reads in whole 8-element chunks
        instead and bounds the descriptor at `ceil8(seqlen_k)` of the last row,
        the same 8xD contract Q/K/V/dQ already live under: the caller's bias row
        stride is a multiple of 8 and at least `ceil8(seqlen_k)`, so the bytes
        between `seqlen_k` and the ceiling are the row's own padding.
        `_args` enforces that. A row past `seqlen_q` is still out of bounds and
        reads zero; a column in `[seqlen_k, ceil8)` reads padding and is masked
        to `-inf` by the KV tail mask that follows the add.
        """
        span = _slab_span_elems(self.seqlen_q_v, self.stride_b_seq_q, self.seqlen_kv_v)
        base_bytes = self._slab_byte_base(
            self.stride_b_batch,
            self.stride_b_head,
            self.stride_b_seq_q,
            self.q_row_off,
            self.q_head_idx,
        )
        self.bias_div = self._slab_view_at(self.Bias, base_bytes, span)

    def bias_lds_base(self):
        """(index, pointer) of the staged bias region. The merged kernel places it behind K/V in its shared arena."""
        bias = self.lds.bias.ptr
        return fx.Index(fx.ptrtoint(bias)), bias.llvm_ptr

    def init_bias_stage(self):
        """Per-lane addressing for the LDS-staged bias tile; a no-op unless `BIAS_STAGE == "lds"`.

        The tile is `BLOCK_M` rows of one 128-byte row each (`BLOCK_N` 64 bf16),
        row `r` at `r * 128` in its own LDS region, in 16-byte chunks. A wave
        owns the `ROWS_PER_WAVE` rows its lanes later read, so it stages exactly
        those: `ROWS_PER_WAVE / 8` DMA issues, each 8 rows (64 lanes x 16 B).
        LDS-DMA writes lane `i` of an issue to `m0 + 16 * i`, i.e. physical slot
        `i % 8` of row `i // 8`; the lane therefore *fetches* global chunk
        `(i % 8) ^ (row & 7)`, which puts chunk `c` of row `r` in slot
        `c ^ (r & 7)`. `row & 7 == i // 8` here because every issue's first row
        is a multiple of 8. The reader undoes it with the same XOR (see
        `BwdDqSoftmaxHelper._add_bias_lds`), which de-conflicts the one-row-per-
        lane `ds_read_b64`s whose rows are 128 B apart.
        """
        traits = self.traits
        if const_expr(traits.BIAS_STAGE != "lds"):
            return
        self.lds_bias_base_idx, self.lds_bias_base_ptr = self.bias_lds_base()
        row_bytes = traits.BLOCK_N * traits.BF16_BYTES
        issues = traits.ROWS_PER_WAVE // 8
        self.bias_dma_m0 = tuple(
            dualwave.rocdl.readfirstlane(
                T.i32,
                as_mlir_value(
                    fx.Int32(
                        self.lds_bias_base_idx
                        + (
                            self.wave_id_uni * fx.Index(traits.ROWS_PER_WAVE)
                            + fx.Index(d * 8)
                        )
                        * fx.Index(row_bytes)
                    )
                ),
            )
            for d in range(issues)
        )
        lane_row = fx.Int32(self.lane_in_warp) // fx.Int32(8)
        chunk = (fx.Int32(self.lane_in_warp) % fx.Int32(8)) ^ lane_row
        row0 = self.q_start + self.wave_q_offset + fx.Index(lane_row)
        self.bias_dma_src = row0 * fx.Index(self.stride_b_seq_q) + fx.Index(
            chunk
        ) * fx.Index(8)

    def init_philox(self):
        """The forward's philox prologue, minus the report.

        **The backward *consumes* the reproducibility contract; it does not
        publish one.** `ParityKernelContext.init_philox` ends by calling
        `fmha.philox_report`, which writes the seed and offset back for a
        later backward to pick up -- correct there, meaningless here, and it
        would need two output pointers on the wire that nothing reads.

        Everything else is inherited verbatim, and that is the whole point:
        the mask has to be regenerated *bit-identically*, and it is because
        both kernels call the same `Philox.grid_plane` with the same
        `(seed, offset)` and the same plane -- not because two transcriptions
        of one formula happened to agree.

        The consequence worth naming: `grid_plane` is given `max_seqlen_q` and
        `max_seqlen_k`, never `BLOCK_M`/`BLOCK_N`, so the mask is a function of
        element coordinates only and **the tile geometry may differ between
        the two kernels**. What may *not* differ is the pair of max lengths or
        the plane index, and the plane index is `(sequence, q head)` -- the raw
        grid `z`, kept as `seq_idx_i32`, and **not** `batch_idx`, which the
        varlen decode resolves to 0 for every sequence of a stacked layout. The
        forward reads the same field for the same reason.
        """
        if const_expr(not self.traits.ENABLE_DROPOUT):
            return
        self.philox_rng = Philox.for_arch("gfx950")
        plane = self.seq_idx_i32 * fx.Int32(self.num_head_q) + fx.Int32(self.q_head_idx)
        seed = fmha.philox_seed_value(self.philox_seed_ptr)
        offset = fmha.philox_offset_base(self.philox_offset1, self.philox_offset2)
        self.philox_seed = seed
        self.philox_plane_base, self.philox_row_stride = self.philox_rng.grid_plane(
            offset, plane, self.seq_len_v, self.seq_len_kv_v
        )

    def compute_active_guard(self):
        """`q_start < seqlen_q`, and **not** the cross-sequence term.

        The base adds `causal_end_raw_i32 > 0` under `CAUSAL and CROSS_SEQLEN`,
        which skips a Q block whose causal region is empty -- correct for the
        forward, which then zeroes `O` for those rows through
        `zero_o_block_if_needed`. Inheriting it here would leave **dQ
        unwritten** for the same rows, which is a garbage read for the caller
        rather than a zero.

        Dropping the term is not a workaround: this loop derives its tile count
        from the same `causal_end_raw_i32`, so an empty causal region walks
        **zero tiles**, leaves the accumulator at its zero seed and stores
        zeros. The block does the right thing by running rather than by being
        skipped, and no second zeroing path is needed in either family.

        The `q_start` term is kept, and it is the one that saves work: the grid
        is sized from `max_seqlen_q`, so a short sequence dispatches Q blocks
        with no real rows at all. A dense call has `seqlen_q_v == max_seqlen_q`
        and the predicate is true for every block the grid dispatches, which is
        why it needs no build-time case -- it is a scalar compare that a dense
        launch simply never fails.
        """
        if const_expr(self.traits.SPLITK):
            return self.split_nonempty
        return self.q_start < self.seqlen_q_v

    def init_tile_bounds(self, **kwargs):
        """The inherited bounds, re-tightened for a **one-tile** loop.

        `ParityKernelContext` resolves the window and re-points `delta_i32`,
        and the base class then derives `max_num_tiles` from it -- all of which
        this body wants. What it does not want is the two lines after: the base
        rounds the count up to even and floors it at 4, because the dual-wave
        pipeline consumes two tiles per iteration and its prologue plus
        epilogue need four to exist. This loop consumes one and needs none.

        Left alone it is still *correct* -- the extra tiles are fully masked and
        contribute nothing -- which is exactly the problem the contract warns
        about: **a dead tile is a no-op, so correctness cannot show the cut
        works.** Up to three dead tiles per Q block is also enough to hide the
        cut from a timing test, which is the only test that can see it.
        """
        super().init_tile_bounds(**kwargs)
        traits = self.traits
        if const_expr(traits.CAUSAL):
            # `causal_end_raw_i32` is `q_start + BLOCK_M + delta_i32`, and
            # `delta_i32` is the resolved *right* bound under a window -- which
            # is why re-pointing it generalises the tile count as well as the
            # mask.
            end_i32 = fx.Int32(
                (self.causal_end_raw_i32 > fx.Int32(0)).select(
                    self.causal_end_raw_i32, fx.Int32(0)
                )
            )
            n = (fx.Index(end_i32) + fx.Index(traits.BLOCK_N - 1)) // fx.Index(
                traits.BLOCK_N
            )
            n = fx.Index((n < self.num_kv_tiles).select(n, self.num_kv_tiles))
        else:
            n = self.num_kv_tiles
        self.max_num_tiles = n
        # **Not a literal 0.** P3 found four literal tile bases in the forward,
        # and a window moves this one: `_skip_dead_leading_tiles` below is the
        # only thing that writes it, and spelling the non-window arm as 0 here
        # would be correct and would throw the left-bound saving away.
        self.split_t0 = fx.Index(0)
        self.split_t_end = n
        if const_expr(traits.WINDOW):
            self._skip_dead_leading_tiles()

    def _skip_dead_leading_tiles(self):
        """Start the KV walk at the window's left edge. Exact, for a one-tile loop.

        The inherited version rounds the base down to even and caps it at
        `split_t_end - 4`, both because the dual-wave pipeline needs an even
        segment with four tiles in it. Neither applies here, and both would
        blunt the cut -- the cap in particular pins the base to zero for any Q
        block whose live range is short, which is the case a left bound is for.

        The lowest column any row of this Q block can reach is
        `q_start - window_left`, since `q_start` is the smallest row. Clamped
        before the divide rather than after: `fx.Index` is unsigned, so a
        negative would come out enormous and skip the whole range.
        """
        traits = self.traits
        first_col_i32 = fx.Int32(self.q_start) - self.window_left_i32
        first_col_i32 = fx.Int32(
            (first_col_i32 > fx.Int32(0)).select(first_col_i32, fx.Int32(0))
        )
        t0 = fx.Index(first_col_i32) // fx.Index(traits.BLOCK_N)
        self.split_t0 = fx.Index((t0 < self.split_t_end).select(t0, self.split_t_end))

    def init_row_inputs(self):
        """Descriptors and the shared row addressing for LSE and delta.

        **One addressing for both**, which is `fmha.lse_row_addressing`'s
        stated contract: delta is produced beside LSE by the same caller and
        giving it its own decode would double the work for no expressiveness.
        Called with batch 0 because the descriptors below already fold the
        batch in, exactly as the forward's varlen LSE store does.
        """
        tokens = fx.Index(self.lse_tokens_i32)
        per_batch = fx.Index(self.num_head_q) * tokens
        per_batch_bytes = per_batch * fx.Index(4)
        self.row_base, self.row_pitch = fmha.lse_row_addressing(
            self.varlen_bits_arg,
            fx.Index(0),
            self.q_head_idx,
            self.num_head_q,
            tokens,
            self.q_row_off,
        )
        # The sentinel a row past `seqlen_q` is redirected to. Without it the
        # offset would run into the *next head's* rows, which is in bounds and
        # returns a plausible LSE for the wrong row -- finite, and wrong in a
        # way no shape check sees. The row is discarded at the store either
        # way, but a zero keeps the arithmetic in between boring.
        self.row_oob_off = per_batch
        self.lse_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))),
            self.batch_idx * per_batch_bytes,
            per_batch_bytes,
        )
        self.delta_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.Delta))),
            self.batch_idx * per_batch_bytes,
            per_batch_bytes,
        )

    def row_delta(self, q_row, off_i32):
        """This row's `Delta`, as the workspace holds it. The merged kernel computes it in place from O and dO instead."""
        return buffer_ops.buffer_load(
            self.delta_rsrc, off_i32, vec_width=1, dtype=fx.Float32
        )

    def row_lse(self, lse):
        """This row's raw `LSE` before the log2 conversion. The merged kernel maps -inf to +inf here (the prep's job)."""
        return lse

    def load_row_scalars(self, q_row):
        """`(lse2, delta)` for one query row. Per-lane scalars in both families.

        The query row is the *n* axis of GEMM1 and GEMM3, so a lane owns
        exactly one of them whatever the MFMA's row count -- which is why this
        is shape-independent and the dK/dV kernel's counterpart is not (there
        the query row is *m*, and a lane holds four).
        """
        off = self.row_base + q_row * self.row_pitch
        off = fx.Index((q_row < self.seqlen_q_v).select(off, self.row_oob_off))
        off_i32 = as_mlir_value(fx.Int32(off))
        lse = buffer_ops.buffer_load(
            self.lse_rsrc, off_i32, vec_width=1, dtype=fx.Float32
        )
        delta = self.row_delta(q_row, off_i32)
        lse = self.row_lse(lse)
        # The forward writes LSE in natural units (`m_row*ln2 + ln(l)`), and
        # every exponent downstream is `exp2`, so the conversion happens once
        # here rather than per element. AOTriton's `l_i = tl.load(...) *
        # RCP_LN2` is the same fold.
        lse2 = _fmul(fx.Float32(lse), fx.Float32(dualwave._LOG2E), self.fm_fast)
        # **Floored, because `LSE` can legitimately be `-inf`** and this is an
        # *input*. A causal row with no live key -- every row below
        # `seqlen_q - seqlen_kv` under bottom-right alignment, which is every
        # `Sq > Sk` varlen sequence -- makes the forward's `l_row` zero and its
        # `m_row*ln2 + log(l_row)` therefore `-inf`. Our own forward writes it.
        #
        # **This is not fixing an observed NaN, and saying so matters.**
        # Measured with the floor removed, the varlen causal oracle still
        # passes bitwise: `scale_and_sub_lse` computes `fma(S, qk_scale,
        # +inf)`, and the KV/causal mask that runs *after* it overwrites every
        # column of such a row with `-inf` before `exp2` sees it. The rescue is
        # real and it is why the mask is ordered after the scale.
        #
        # The floor is kept because that rescue leans on an infinity flowing
        # through an FMA carrying `fastmath<fast>`, whose `ninf` is a licence
        # to assume infinities are absent. P5 recorded exactly that licence
        # being taken up later by a different pass and silently deleting a KV
        # tail mask on gfx1201. One `max` per kernel removes the case.
        #
        # After the `log2e` conversion, not before: `-3e38 * log2e` overflows
        # back to `-inf`. Same device `ParitySoftmaxHelper` uses to keep the
        # forward's running max off `-inf`, applied to an input.
        #
        # **Re-derived for f16 rather than assumed to carry.** The floor is
        # `-3.0e38`, which is an f32 magnitude and would be `-inf` in f16 --
        # but nothing here is f16. `lse` arrives f32, `log2e` is f32, the `max`
        # is f32, and the FMA it protects in `scale_and_sub_lse` is f32. The
        # operand dtype reaches this kernel at exactly two places, the `dS`
        # pack and the dQ store, and neither is upstream of this. So the
        # argument is about `fastmath<fast>`'s `ninf` on an f32 FMA and is
        # dtype-independent.
        lse2 = _fmax(fx.Float32(lse2), self.c_neg_floor, self.fm_fast)
        return lse2, fx.Float32(delta)


class BwdRowInputLoader(ParityStoreHelper):
    """`lse2` and `delta` for this lane's query row, once per kernel.

    A thin wrapper on `BwdDqKernelContext.load_row_scalars`, kept because the
    32-row body reaches it through a helper object like everything else. The
    work is on the context because the 16-row body needs the same two values
    at a different row map (`lane % 16`), and one copy of the addressing is
    the point.
    """

    def load(self, q_row):
        return self.ctx_ref.load_row_scalars(q_row)


class BwdDqSoftmaxHelper(ParitySoftmaxHelper):
    """The forward's softmax, minus the one place it trades accuracy for speed."""

    # -- LDS-staged bias: where the bias tile comes from ------------------------
    #
    # `ParitySoftmaxHelper.seq_pad_mask_if_needed` calls `_add_bias_inplace` on
    # every tile, ahead of the KV tail mask; overriding it here swaps the source
    # of the bias and nothing else, so the add, its order against the mask and
    # the `log2e` scaling are the incumbent's.

    def _add_bias_inplace(self, v_s, tile_idx, narrow_tail=False):
        if const_expr(self.traits.BIAS_STAGE == ""):
            return super()._add_bias_inplace(v_s, tile_idx, narrow_tail=narrow_tail)
        s_lo, s_hi = v_s
        log2e = fx.Float32(1.4426950408889634)
        # `contract | reassoc` and not `fm_fast`, for the incumbent's reason: a bias of
        # `-inf` is a real infinity going into arithmetic.
        fm_bias = (
            dualwave.arith.FastMathFlags.contract | dualwave.arith.FastMathFlags.reassoc
        )
        runs = _score_column_runs(self.traits.KV_VECTORIZED)
        for half, values in ((0, s_lo), (1, s_hi)):
            for elem0, col_off, width in runs:
                span = self._bias_lds_run(half, col_off, width)
                for j in range_constexpr(width):
                    b = fx.Float32(
                        Vec(span, (width,), self.elem_dtype)[j].to(fx.Float32)
                    )
                    values[elem0 + j] = _fadd(
                        values[elem0 + j], _fmul(b, log2e, fm_bias), fm_bias
                    )

    def _bias_lds_run(self, half, col_off, width):
        """This lane's `width` bias elements of one score run, out of the staged tile.

        The lane's row is `q_row_in_block` (rows are `ROWS_PER_WAVE` per wave and
        the wave stages its own), 128 B apart; the run starts at tile column
        `half * 32 + col_off + 4 * lane_div_32`, which is 16-byte chunk
        `half * 4 + col_off / 8` at byte `8 * lane_div_32` within it. The chunk
        sits in physical slot `chunk ^ (row & 7)` (`init_bias_stage`), and
        `row & 7 == lane_mod_32 & 7` because a wave's first row is a multiple of
        32. One `ds_read_b64`.
        """
        traits = self.traits
        ctx = self.ctx_ref
        assert width * traits.BF16_BYTES == 8 and col_off % 8 == 0
        row_bytes = traits.BLOCK_N * traits.BF16_BYTES
        chunk = half * 4 + col_off // 8
        swz = (fx.Int32(self.lane_mod_32) & fx.Int32(7)) * fx.Int32(16)
        byte = (
            fx.Int32(self.q_row_in_block) * fx.Int32(row_bytes)
            + fx.Int32(self.lane_div_32) * fx.Int32(8)
            + (swz ^ fx.Int32(chunk * 16))
        )
        ptr = buffer_ops.get_element_ptr(
            ctx.lds_bias_base_ptr, byte_offset=as_mlir_value(byte), elem_type=T.i8
        )
        return llvm.LoadOp(
            Vec.make_type(width, self.elem_dtype), ptr, alignment=8
        ).result

    def scale_and_sub_lse(self, v_s, qk_scale, lse2):
        """`qk_scale * S - lse2`, one FMA per score element.

        **This is where the backward stopped matching the forward, and it was
        worth 10x at a large `sm_scale`.** The forward folded
        `sm_scale * log2e` into Q and rounded the product back to bf16, which
        saves a multiply per score; the error that introduces is `|S| * 2^-8`
        in the *exponent*, so `P = exp2(S - lse2)` inherits it as a relative
        error. The forward was thought to tolerate that because `O` is a
        normalised average and the error largely cancels, while
        `dS = P * (dP - delta)` does not normalise and `dQ` sums it over the
        whole key axis. **The forward did not tolerate it either** -- the
        measurement below transfers, and `ParityQLoader.scale_all` records what
        it cost there -- so the fold is gone from both and the table is now the
        history of a fixed bug rather than a live divergence.

        Measured at `B=1 H=4 S=512 d=64`, max error against an fp64 reference:

            sm_scale   Q pre-scaled (the old fold)     scaled here
              0.05            1.8e-4                     1.6e-4
              0.25            4.3e-2                     2.0e-2
              1.00            6.9e-1                     6.8e-2

        A host model of both variants reproduces the kernel's own numbers to
        two digits, which is what identifies the Q rounding rather than
        anything else as the cause. AOTriton scales after the dot in *both*
        directions (`qk += Qk_scale * tl.dot(q, k)`), so this is its
        arithmetic, not a new choice.

        Not `dualwave._scale_sub_score_pair`: that one derives the offset from
        an *unscaled* row max (`fma(s, scale, -scale*m)`), and `lse2` is
        already in the scaled base-2 domain. Passing it there would need a
        divide by `sm_scale` to undo a multiply.

        The KV tail mask runs **after** this, not before, so no infinity ever
        reaches the FMA -- `qk_scale * -inf` is correct but puts a real
        infinity into arithmetic under `fastmath<fast>`, which plan section 5
        records as the thing that silently deleted a mask on gfx1201.
        """
        s_lo, s_hi = v_s
        acc = self.traits.ACC_ELEMS
        scale_v = Vec.from_elements([fx.Float32(qk_scale)], fx.Float32).broadcast_to(
            acc
        )
        neg_lse2 = _fsub(self.c_zero_f, lse2, self.fm_fast)
        neg_v = Vec.from_elements([fx.Float32(neg_lse2)], fx.Float32).broadcast_to(acc)
        lo = fmath.fma(Vec(s_lo), scale_v, neg_v, fastmath=self.fm_fast)
        hi = fmath.fma(Vec(s_hi), scale_v, neg_v, fastmath=self.fm_fast)
        return as_mlir_value(lo), as_mlir_value(hi)

    def cast_p(self, v_p, tile_idx=None):
        """Pack to bf16 with **no dropout**, unlike the forward's `cast_p`.

        `ParitySoftmaxHelper.cast_p` is where the *forward* applies its mask --
        after the row sum, before the O accumulation. Inheriting it here would
        drop `dS` on the way into GEMM3 while `dropout_dp` has already dropped
        `dP`, i.e. **apply the mask twice**: once correctly and once to a
        quantity that must not carry it, since `P` is the undropped softmax
        the undropped `lse` defines.

        It fails loudly today only by accident -- the forward's block reads
        `tile_idx`, which this call site does not pass. That is a bad reason to
        be safe, so the override is explicit.

        **Every pack leaves through `mfma_operand_wait_state`, and here that is
        load-bearing rather than prophylactic.** `dS` is GEMM3's B operand, and
        at `BLOCK_DMODEL=64, PADDED_HEAD=False, bf16` the scheduler put all
        four packs in one register quad and issued the *first* `D_CHUNKS`
        accumulator's MFMA in the slot right after the last
        `v_cvt_pk_bf16_f32` of each pack -- one intervening vector-pipe
        instruction at the best site and zero at the worst:

            v_cvt_pk_bf16_f32 v33, v42, v43
            s_waitcnt lgkmcnt(2)                        <- scalar, no wait state
            v_mfma_f32_32x32x16_bf16 v[0:15], v[52:55], v[32:35], v[0:15]

        The second chunk's MFMA reads the same quad one slot later and is
        shielded by the first one, so the damage is confined to head-dim
        columns `[0, 32)`: measured 39% relative L2 on dQ against 7% on
        `[32, 64)`, deterministic, dQ only, and invisible in an f16 build,
        whose `v_cvt_f16_f32` packs the scheduler happened to hoist 8-36
        instructions clear. That is the `v_cvt_pk_bf16_f32` -> MFMA hazard with
        a wrong answer attached rather than a suspicion, and
        `fmha_bwd_dq_m16_gfx950.pack_ds` already carried the barrier on the
        strength of the suspicion alone.
        """
        p_lo_packs, p_hi_packs = dualwave.DualwaveSoftmaxHelper.cast_p(self, v_p)
        return (
            [mfma_operand_wait_state(p) for p in p_lo_packs],
            [mfma_operand_wait_state(p) for p in p_hi_packs],
        )

    def exp2(self, v_s, start, length):
        """`_exp2_score_slice` with the gfx950 `v_exp_f32` wait state.

        The body is `dualwave._exp2_score_slice` verbatim apart from the one
        call, and copied rather than wrapped for the reason the forward's
        equivalent gives: on the `start == 0` path the results are consumed
        *inside* it, by `Vec.from_elements`, so there is no return value a
        wrapper could interpose on.

        See `exp2_wait_state` for why a bare `_s_nop` is not this.

        **Not reproduced in dQ, and imported on the argument.** The same scan
        measured the gap from each `v_exp_f32` to its first VALU consumer over
        ten dQ builds: the minimum anywhere was 8, with per-build minima of 8,
        9, 26, 50 and 54, against the *exactly 1* that AOTriton's dK/dV scan
        reports and that phase 1 confirmed here. dQ is not near this cliff, and
        saying otherwise would overstate it.

        Imported anyway, because "not near the cliff" is a fact about one
        schedule and not about the kernel. AOTriton's live case became live
        exactly that way -- issue 9's fix raised register pressure, the
        schedule moved, and a latent gap became a zero-gap wrong answer. The
        cost here is measured rather than assumed; see the plan's outcome
        section for the register and ISA comparison.
        """
        if const_expr(start == 0):
            s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            lo_partial = exp2_wait_state(
                [
                    dualwave.rocdl.exp2(T.f32, as_mlir_value(s_lo[r]))
                    for r in range_constexpr(16)
                ]
            )
            return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]
        lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        hi_full = exp2_wait_state(
            [
                dualwave.rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r]))
                for r in range_constexpr(16)
            ]
        )
        return lo_partial, hi_full

    def dropout_dp(self, dp_lists, tile_idx, q_row):
        """`dP <- keep ? dP * (1/(1-p)) : 0`, on the **dP** rather than on P.

        AOTriton's `bwd_inner_dq` applies the mask here and this follows it,
        because that is what the chain rule says: the forward's output is
        `sum_j (P_j * keep_j * scale) V_j`, so the gradient with respect to the
        *undropped* `P_j` carries the same `keep_j * scale` factor. `P` itself
        stays undropped -- it is `exp2(qk_scale*S - lse2)` and `lse2` is the
        undropped logsumexp the forward wrote.

        **`delta` needs no adjustment**, and that is not luck:
        `delta = rowsum(dO * O)` with `O` the *dropped* output is exactly
        `sum_j P_j keep_j scale (dO . V_j)`, which is the reduction term the
        softmax backward wants against a masked `dP`. The host already computes
        it from the forward's own `O`.

        **The survivor scale cannot be folded the way the forward folds it.**
        There it becomes one multiply on `1/l` per output row, because a row
        sum exists to fold into; here the row sum is an *input*. Folding it
        into `dO` before GEMM2 would be one multiply instead of one per
        element -- and would round `dO` through bf16 a second time, which is
        the mistake B2 measured at 10.9x on the Q pre-scale. It stays an f32
        multiply on the scores.

        The column runs are the bias and forward-dropout ones: a lane's 16
        scores are a few contiguous spans starting at multiples of
        `randoms_per_offset`, so each is a whole number of Philox calls with no
        partial draw.
        """
        traits = self.traits
        ctx = self.ctx_ref
        rng = ctx.philox_rng
        lo, hi = dp_lists
        lane_n_off = 8 if traits.KV_VECTORIZED else 4
        col_base = fx.Int64(tile_idx * traits.BLOCK_N) + fx.Int64(
            self.lane_div_32 * fx.Index(lane_n_off)
        )
        scale = fx.Float32(ctx.dropout_scale_arg)
        zero = fx.Float32(0.0)
        for half, values in ((0, lo), (1, hi)):
            for elem0, col_off, width in _score_column_runs(traits.KV_VECTORIZED):
                first = rng.grid_offset(
                    ctx.philox_plane_base,
                    ctx.philox_row_stride,
                    fx.Int64(q_row),
                    col_base + fx.Int64(col_off + half * 32),
                )
                keep = rng.keep_span(ctx.philox_seed, first, width, ctx.idropout_p)
                for j in range_constexpr(width):
                    kept = _fmul(values[elem0 + j], scale, self.fm_fast)
                    values[elem0 + j] = keep[j].select(fx.Float32(kept), zero)
        return (lo, hi)


class BwdSecondaryQLoader(ParityQLoader):
    """A second Q-shaped loader, for dO.

    `ParityQLoader` reads three attributes to find its tensor -- the buffer
    view, the sequence stride and the tile's element origin -- and everything
    else about it is the lane map, which dO shares with Q exactly. So this
    rebinds those three and inherits the loop, rather than being a second copy
    of an addressing scheme.

    `hdim_qk` is rebound too: the padded-head mask inside `load_pack` is
    against the *reduction* extent for Q, and dO's is `hdim_vo` (it contracts
    with V's D axis, not K's). Inert while `PADDED_HEAD` is off, which is B2 --
    set correctly now so B3 does not have to find it.
    """

    def __init__(self, ctx, div, stride_seq_v, gmem_elem_offset, hdim):
        super().__init__(ctx)
        self.q_div = div
        self.stride_q_n_v = stride_seq_v
        self.q_gmem_elem_offset = gmem_elem_offset
        self.hdim_qk = hdim


class BwdDqKvGmemToLdsLoader(ParityKvGmemToLdsLoader):
    """One staging routine, two (tensor, region) pairs.

    The inherited `load_k` / `load_v` pair hardcodes which tensor goes to which
    region, and this kernel puts **K in the V region**. `_stage` is the
    inherited body with the tensor, the stride and the `m0` table all handed
    in, so the two calls below differ only in their arguments.

    The `m0` table *is* the pitch: `k_dma_m0` addresses lines at
    `SMEM_K_LINE_STRIDE` and `v_dma_m0` at `SMEM_V_LINE_STRIDE`, and each has to
    match the reader that later walks the region. Pairing them wrongly reads a
    tile written with a different pitch -- plausible numbers, no fault.

    Setting `stride_kv_n_v` before delegating is the inherited trick and is
    safe for the inherited reason: tracing is eager, so the attribute is read
    while the call below runs and no branch is open across the swap.
    """

    def _stage(self, tile_start, src_div, stride_seq_v, dma_m0, buf_id):
        self.stride_kv_n_v = stride_seq_v
        self.dma_stage = 0
        self._async_load_kv_linear(
            dma_m0,
            buf_id,
            src_div,
            self.kv_gmem_elem_offset,
            tile_start * stride_seq_v,
            self.NUM_DMA_K,
        )

    def stage_k(self, tile_start):
        """K into the V region: GEMM1's and GEMM3's A operand, one copy."""
        self._stage(
            tile_start, self.k_div, self.stride_k_seq_v, self.ctx_ref.v_dma_m0, 0
        )

    def stage_v(self, tile_start):
        """V into the K region: GEMM2's A operand."""
        self._stage(
            tile_start, self.v_div, self.stride_v_seq_v, self.ctx_ref.k_dma_m0, 0
        )

    def stage_bias(self, tile_idx):
        """`BIAS_STAGE == "lds"`: this wave's `ROWS_PER_WAVE` bias rows of KV tile `tile_idx`, into the bias region.

        The K/V DMA primitive (`_issue_kv_dma`: one 16-byte `buffer_load ... lds` per lane,
        `m0` the wave-uniform LDS base) with the bias's addressing in place of
        `kv_src_elem`'s token/d map, which indexes a `[token][d]` tile. Issue `d`
        covers rows `8d .. 8d+7` of the wave's rows: the 8 lanes of a row read one
        contiguous 128 B of the bias row, fully coalesced. `soffset` is the tile's
        first column in elements. Covered by the iteration's existing
        `s_waitcnt(0)` + `s_barrier`; no extra round trip.
        """
        ctx = self.ctx_ref
        traits = self.traits
        soffset = tile_idx * fx.Index(traits.BLOCK_N)
        for d in range_constexpr(traits.ROWS_PER_WAVE // 8):
            self._issue_kv_dma(
                ctx.bias_div,
                ctx.bias_dma_m0[d],
                ctx.bias_dma_src + fx.Index(d * 8) * fx.Index(ctx.stride_b_seq_q),
                soffset,
            )


class BwdDqKvLdsToVgprLoader(ParityKvLdsToVgprLoader):
    """The forward's two readers, with the K path re-pointable at the V pitch.

    One instance per tile, because the *padded-head extent* differs between
    them as much as the pitch does: the K tile's D axis is `hdim_qk` and the V
    tile's is `hdim_vo`. B2 had one instance and would have masked V's columns
    against Q's extent the moment a padded head arrived -- one of the two
    crossed sites the plan's B2 outcome names.

    `v_layout` is a Python bool fixed at construction, so every `const_expr`
    below folds and neither instance emits the other's code.
    """

    def __init__(self, ctx, *, v_layout, hdim, hdim_floor):
        super().__init__(ctx)
        self.v_layout = bool(v_layout)
        # `load_k`'s mask reads these two by these names. Rebinding them is
        # what makes one class serve both extents.
        self.hdim_qk = hdim
        self.HDIM_QK_FLOOR = int(hdim_floor)

    def _lds_pack(self, elem_idx, scope_name):
        """One 8xbf16 MFMA operand pack out of LDS, under a chosen alias scope.

        `dualwave._load_k_pack_aligned` is the same four lines, except that it
        derives the scope as `lds_k{buf_id}` -- and this kernel's K tile lives
        in the `v0` region. A scope that names the wrong region is not a
        cosmetic problem: `ROCDL_LDS_Read_Tr_IntrOp` and this load are told
        they cannot alias, and here they read the same bytes.
        """
        traits = self.traits
        ptr = buffer_ops.get_element_ptr(
            self.lds_kv_base_ptr,
            byte_offset=elem_idx * traits.BF16_BYTES,
            elem_type=T.i8,
        )
        return llvm.LoadOp(
            self.kv_mfma_pack_type,
            ptr,
            alignment=16,
            alias_scopes=dualwave._dualwave_lds_alias_scopes(scope_name),
            noalias_scopes=dualwave._dualwave_lds_noalias_scopes(
                scope_name, traits.LDS_SCOPE_NAMES
            ),
        ).result

    def _read_k_packs(self, buf_id, urk_base):
        """The K path, over whichever region this instance owns.

        The `v_layout` arm re-points three things and no more: the line pitch,
        the k-step outer stride (which is `SMEM_N_RPT` lines) and the region
        base. Everything else -- `_parity_k_read_base`, `_parity_ks_offset`,
        the `N_STRIP` lo/hi split -- is the shared formula, evaluated against a
        `replace`d traits object rather than transcribed. The two pitches are
        the *only* difference between the layouts, which is what makes the
        substitution total.

        `urk_base` is ignored on this arm: `load_k` computes it from
        `k_lds_read_base_per_lane`, which the context derived at the K pitch.
        """
        if const_expr(not self.v_layout):
            return super()._read_k_packs(buf_id, urk_base)
        full = self.traits
        self.traits = replace(
            full,
            SMEM_K_LINE_STRIDE=full.SMEM_V_LINE_STRIDE,
            K_LDS_TO_REG_KSTEP_OUTER_STRIDE=full.SMEM_N_RPT * full.SMEM_V_LINE_STRIDE,
        )
        try:
            traits = self.traits
            scope = dualwave._dualwave_lds_scope("v", buf_id)
            base = dualwave._v_buf_base(traits, buf_id) + _parity_k_read_base(
                traits, self.lane_mod_32, self.lane_div_32
            )
            k_lo = [None] * traits.K_STEPS_QK
            k_hi = [None] * traits.K_STEPS_QK
            for ks in range_constexpr(traits.K_STEPS_QK):
                idx = base + _parity_ks_offset(traits, ks)
                k_lo[ks] = self._lds_pack(idx, scope)
                k_hi[ks] = self._lds_pack(
                    idx + traits.K_LDS_TO_REG_N_STRIP_STRIDE, scope
                )
            return k_lo, k_hi
        finally:
            self.traits = full


class BwdDqStoreHelper(ParityStoreHelper):
    """The inherited store, pointed at dQ.

    Upstream's `DualwaveStoreHelper` has exactly one output vocabulary and it
    is spelled `o_*`: `o_div` is the descriptor it writes through,
    `stride_o_seq_v` the row pitch it addresses with, `o_oob_off` the sentinel
    it redirects suppressed chunks to, and `hdim_vo` the width it suppresses
    past. This kernel's output is dQ, so those four names are bound to dQ's
    here -- in one block, at the boundary with the code that owns them. The
    context below spells the tensor `DQ`, its strides `stride_dq_*` and its
    descriptor `dq_div`, so nothing outside this block has to know that the
    inherited store calls it O.

    `hdim_qk`, not `hdim_vo`, for the width: dQ is Q-shaped. The two coincide
    in every symmetric build and cross the moment they do not -- the second of
    the two sites the plan's B2 outcome names.

    Rebinding attributes rather than overriding methods: the store helper is a
    dict copy of the context, and the suppression is one comparison inside a
    method that also computes the address, so a copy of it would be a copy of
    both.
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        self.o_div = ctx.dq_div
        self.o_oob_off = ctx.dq_oob_off
        self.stride_o_seq_v = ctx.stride_dq_seq_v
        self.hdim_vo = ctx.hdim_qk


class BwdDbStoreHelper(ParityStoreHelper):
    """`dB = dS` for one KV tile, one element per store.

    **Per element, and that is a correctness decision rather than an
    oversight.** A lane's 16 scores are four runs of four contiguous KV
    columns (`_score_column_runs`), so a 4-wide vector store is available and
    is what the bias *load* uses. It is not available here: a run straddling
    `seqlen_k` would have to be partially written, and a 128-bit store is
    all-or-nothing. Suppressing the whole run instead is exact only when
    `seqlen_k` is a multiple of 4, and silently corrupting the next row of dB
    when it is not is precisely the failure mode this phase is written to
    avoid. So the tail is paid for in full, on a path that is off by default.

    Suppression is by *address*, not by a branch -- pushing the offset past the
    descriptor's `num_records` makes the hardware drop the store -- which is
    the same device `ParityStoreHelper._final_o_global` and `_store_lse_row`
    use.

    It costs 2-5.5x, measured across the ladder, and it is a *cost* rather than
    a ceiling: every rung including 512 builds and is correct with it on. The
    numbers and the register accounting are in the B3 outcome section of
    `sdpa-bwd-plan-gfx950.md`. A vectorised version needs *two* things and not
    one: a runtime second arm for the tile containing `seqlen_k`, and a
    `stride_db_seq_q` divisible by 4, since an 8-byte store off a row pitch of,
    say, 201 elements is 2-byte aligned.

    --- The source is the packed bf16, not the f32 -----------------------------

    **This shortens a live range rather than shrinking a value**, and at the
    top of the ladder that is the difference that matters. `dS` exists twice:
    as 32 f32 scores, and -- after `cast_p` -- as 4 packed `v8` bf16 vectors,
    16 VGPRs against 32. Storing from the f32 list keeps *both* alive across
    the 32-store sequence and its address arithmetic, at exactly the point
    where head_dim 384 and 512 have no registers left. Reading from the packs
    lets the f32 form die at `cast_p`.

    The values are identical: `dB` is a bf16 tensor either way, so the f32 ->
    bf16 rounding happens once regardless -- this only moves *where*. The
    element map is unchanged, because `_pack_p_v8_slices` packs list element
    `pks*8 + j` into lane slot `j` of pack `pks`, in order.
    """

    def store_tile(self, ds_packs, tile_idx, q_row):
        traits = self.traits
        ctx = self.ctx_ref
        lo_packs, hi_packs = ds_packs
        row_base = q_row * fx.Index(ctx.stride_db_seq_q)
        col_base = dualwave._seq_pad_col_base(
            traits, tile_idx, lane_div_32=self.lane_div_32
        )
        # `db_store_enabled` folded in here, once per tile, rather than into
        # `live` once per element: it is uniform, and the per-element form would
        # cost an instruction per score on the rungs that have no registers to
        # spare. A disabled store lands on `db_oob_off` and the hardware drops it,
        # which is the suppression device this whole helper is built on.
        in_row = (q_row < self.seqlen_q_v) & ctx.db_store_enabled
        per_pack = traits.OPERAND_LANE_ELEMS
        for half, packs in ((0, lo_packs), (1, hi_packs)):
            for pks in range_constexpr(len(packs)):
                vec = Vec(packs[pks], (per_pack,), self.elem_dtype)
                for j in range_constexpr(per_pack):
                    # The same element -> column map the KV tail mask uses.
                    # Derived once, in `flash_attn_utils`, and read here rather
                    # than transcribed: the mask and this store must agree
                    # about which column an element is, or dB is a permutation
                    # of itself.
                    r = pks * per_pack + j
                    col_i32 = col_base + fx.Int32(
                        dualwave._seq_pad_score_threshold(traits, r)
                        + half * traits.MFMA_M
                    )
                    live = in_row & (col_i32 < self.seqlen_kv_i32)
                    off = fx.Index(
                        live.select(row_base + fx.Index(col_i32), ctx.db_oob_off)
                    )
                    buffer_ops.buffer_store(
                        as_mlir_value(vec[j]), ctx.db_rsrc, as_mlir_value(fx.Int32(off))
                    )

    def store_tile_m16(self, ds_pack, tile_idx, q_row, lane):
        """`dB = dS` for the 16-row family.

        Same device as `store_tile` -- per element, address-suppressed, read
        out of the packed bf16 so the f32 form can die -- over the 16-row
        column map instead of `_seq_pad_score_threshold`'s. A lane's four
        accumulator elements are four *contiguous* KV tokens at
        `4 * (lane // 16)`, so the map is an addition rather than a table.

        Slot `4r + i` of the pack is half `r`, element `i`, which is the same
        concatenation GEMM3's B operand relies on: if this indexing is wrong,
        dB is a permutation of itself and dQ is simply wrong, so the two check
        each other.
        """
        from kernels.attention_bwd_cdna.fmha_bwd_dq_m16_gfx950 import (
            MFMA16_M,
            acc_elems,
        )

        traits = self.traits
        ctx = self.ctx_ref
        n = acc_elems(traits)
        row_base = q_row * fx.Index(ctx.stride_db_seq_q)
        in_row = (q_row < self.seqlen_q_v) & ctx.db_store_enabled  # see store_tile
        tok_lane = fx.Int32(4) * fx.Int32(lane // fx.Index(MFMA16_M))
        vec = Vec(ds_pack, (n * 2,), self.elem_dtype)
        for r in range_constexpr(2):
            for i in range_constexpr(n):
                col_i32 = (
                    fx.Int32(tile_idx * traits.BLOCK_N)
                    + fx.Int32(r * MFMA16_M + i)
                    + tok_lane
                )
                live = in_row & (col_i32 < self.seqlen_kv_i32)
                off = fx.Index(
                    live.select(row_base + fx.Index(col_i32), ctx.db_oob_off)
                )
                buffer_ops.buffer_store(
                    as_mlir_value(vec[r * n + i]),
                    ctx.db_rsrc,
                    as_mlir_value(fx.Int32(off)),
                )


def build_fmha_bwd_dq_gfx950_module_primary(meta, knobs, runtime_axes=()):
    """Build the backward dQ kernel for a resolved (meta, knobs) pair.

    Same shape as the forward's builder and for the same reason: `meta` is what
    the caller asked for, `knobs` is what the tuning policy answered, and
    nothing here falls back to a policy.
    """
    # `runtime_axes` ⊆ {"padded", "dropout"}: that axis becomes a launch-uniform branch on `rt_flags`
    # (bit0 padded, bit1 dropout) selecting one of two whole bodies; geometry comes from `knobs` unchanged.
    runtime_axes = frozenset(runtime_axes)
    assert runtime_axes <= {"padded", "dropout"}, runtime_axes
    if knobs.block_dmodel is None:
        raise ValueError(
            "knobs must be resolved: call `bwd_dq_knobs(arch, ...).resolve(meta)` first"
        )
    traits = knobs.build_traits(meta)

    BLOCK_DMODEL = knobs.block_dmodel
    PADDED_HEAD = knobs.padded_head
    HDIM_QK_FLOOR = knobs.hdim_qk_floor
    # f32 accumulator elements per lane per MFMA -- 16 at 32x32x16. Every
    # `16` the body used to spell over a score half was this, and each was a
    # different quarter of the kernel. See `BwdDqTraits.ACC_ELEMS`.
    ACC_ELEMS = traits.ACC_ELEMS
    STORE_DB = traits.STORE_DB
    LDS_BIAS = traits.BIAS_STAGE == "lds"
    # Which MFMA family this build is. `MFMA_N` is the query rows a wave owns,
    # so it is the discriminator rather than a width threshold -- the same
    # shape of choice `WIDE` is in the forward.
    M16 = traits.MFMA_N == 16
    BUILD_SM_SCALE = meta.sm_scale

    # Everything the build depends on, so that two builds differing anywhere
    # cannot collide in the JIT disk cache -- which a knob sweep hits first.
    # `traits_cache_key` is every trait field rather than `traits.cache_tag`'s
    # subset; see there for what the subset was silently getting wrong. It
    # subsumes `STORE_DB`, `HDIM_VO_FLOOR` and the MFMA shape, which this tag
    # used to name individually and which are all `traits.` reads.
    #
    # `strides_constexpr` is **new** here and was a hole of the same kind:
    # `abi.varlen_args` below reads it, so it changes the emitted kernel, and
    # it appeared in neither half of the old key. The forward and dK/dV tags
    # both had it.
    _cache_tag = (
        traits_cache_key(traits),
        BLOCK_DMODEL,
        PADDED_HEAD,
        HDIM_QK_FLOOR,
        knobs.strides_constexpr,
        BUILD_SM_SCALE,
        tuple(sorted(runtime_axes)),
    )

    _lds_elem_dtype = dualwave.dtype_to_elem_type(traits.DTYPE_STR)

    if traits.BIAS_STAGE == "lds":

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
            bias: fx.Array[
                _lds_elem_dtype, traits.BIAS_LDS_BYTES // traits.BF16_BYTES, 16
            ]

    else:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.jit
    def emit_dq_body(a, ctx_cls=BwdDqKernelContext, ctx_kwargs=None):
        """The whole kernel body over one namespace `a` of the wire arguments; the kernel below is `a = args; emit`.

        `ctx_cls` is the context class (the merged kernel substitutes a subclass) and `ctx_kwargs`
        are extra context constructor arguments. Defaults reproduce this kernel exactly.
        """
        ctx_kwargs = {} if ctx_kwargs is None else ctx_kwargs
        Q = a.Q
        K = a.K
        V = a.V
        B = a.B
        DO = a.DO
        DQ = a.DQ
        DB = a.DB
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
        stride_dq_batch = a.stride_dq_batch
        stride_dq_head = a.stride_dq_head
        stride_dq_seq = a.stride_dq_seq
        stride_b_batch = a.stride_b_batch
        stride_b_head = a.stride_b_head
        stride_b_seq_q = a.stride_b_seq_q
        stride_db_batch = a.stride_db_batch
        stride_db_head = a.stride_db_head
        stride_db_seq_q = a.stride_db_seq_q
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
        DQ = wire_view(DQ)
        DB = wire_view(DB)
        LSE = wire_view(LSE)
        Delta = wire_view(Delta)
        # dQ's three strides, named once. They reach the context twice, and
        # the second of the two is a residue with a retirement condition:
        #
        # - `dq_strides` is the real one. It builds `dq_div`, bounded by
        #   `hdim_qk`, which is what `BwdDqStoreHelper` writes through.
        # - `o_strides` fills the base class's `O` slot, which this kernel has
        #   no tensor for. It is still needed because
        #   `ParityKernelContext.init_descriptors` builds an O view
        #   unconditionally; the view it builds here is dead -- the store reads
        #   `dq_div` -- and folds away, being pure descriptor arithmetic.
        #
        # Retire the second, along with `O=DQ` below, once the shared context
        # grows the `if self.O is not None` guard that dK/dV's split wants
        # anyway. That file has another owner, so this stays a duplicated
        # argument rather than a cross-file change.
        ctx = ctx_cls(
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
            ),
            dq_strides=(stride_dq_batch, stride_dq_head, stride_dq_seq),
            do_strides=(stride_do_batch, stride_do_head, stride_do_seq),
            db_strides=(stride_db_batch, stride_db_head, stride_db_seq_q),
            sm_scale=sm_scale,
            num_head_q=num_head_q,
            num_head_k=num_head_k,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            padded_head=PADDED_HEAD,
            hdim_qk_floor=knobs.hdim_qk_floor,
            window_left=window_left,
            window_right=window_right,
            seqinfo=(seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1),
            varlen_bits=varlen_bits,
            num_seqlens=num_seqlens,
            Q=Q,
            K=K,
            V=V,
            DQ=DQ,
            DO=DO,
            DB=DB,
            Delta=Delta,
            Bias=B,
            bias_strides=(stride_b_batch, stride_b_head, stride_b_seq_q),
            philox=(philox_seed_ptr, philox_offset1, philox_offset2, None, None),
            idropout_p=idropout_p,
            dropout_scale=dropout_scale,
            seq_len=max_seqlen_q,
            seq_len_kv=max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
            block_table_stride=0,
            LSE=LSE,
            **ctx_kwargs,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_philox()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds()
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()
        ctx.init_bias_stage()
        ctx.init_q_row()
        ctx.init_row_inputs()

        _build_traits = traits

        def _arm_traits(drop_on):
            if bool(drop_on) == bool(_build_traits.ENABLE_DROPOUT):
                return _build_traits
            return replace(_build_traits, ENABLE_DROPOUT=bool(drop_on))

        def _make_arm(padded_on, drop_on):
            ctx.PADDED_HEAD = padded_on
            traits = _arm_traits(drop_on)
            ctx.traits = traits
            kv_gmem_to_lds = BwdDqKvGmemToLdsLoader(ctx)
            # Two reader instances, and they differ in *both* the LDS pitch and the
            # padded-head extent. `k_reader` also serves GEMM3 through the stock
            # `load_v(0)`, since that reads the same region it does.
            k_reader = BwdDqKvLdsToVgprLoader(
                ctx, v_layout=True, hdim=hdim_qk, hdim_floor=HDIM_QK_FLOOR
            )
            v_reader = BwdDqKvLdsToVgprLoader(
                ctx, v_layout=False, hdim=hdim_vo, hdim_floor=traits.HDIM_VO_FLOOR
            )
            q_loader = ParityQLoader(ctx)
            do_loader = BwdSecondaryQLoader(
                ctx, ctx.do_div, ctx.stride_do_seq_v, ctx.do_gmem_elem_offset, hdim_vo
            )
            gemm_helper = ParityGemmHelper(ctx)
            softmax_helper = BwdDqSoftmaxHelper(ctx)
            output_store = BwdDqStoreHelper(ctx)
            row_inputs = BwdRowInputLoader(ctx)
            db_store = BwdDbStoreHelper(ctx)

            def _body():
                """One KV tile per iteration; three GEMMs and one accumulator."""
                # Neither operand is pre-scaled: `qk_scale` is applied to the f32
                # scores below, where it costs the same FMA the `lse2` subtract
                # needed anyway and does not round Q through bf16 a second time.
                # See `BwdDqSoftmaxHelper.scale_and_sub_lse`.
                q_all_bf16 = q_loader.load_all()
                do_all_bf16 = do_loader.load_all()
                lse2, delta = row_inputs.load(ctx.q_row)

                init_args = [
                    ctx.c_zero_v16f32 for _ in range_constexpr(traits.D_CHUNKS)
                ]
                loop_results = init_args
                for j, loop_args in range(
                    ctx.split_tile(0), ctx.split_t_end, fx.Index(1), init=init_args
                ):
                    v_dq = _carried(loop_args, traits.D_CHUNKS)
                    tile_start = ctx.tile_start(j)

                    # Closes the previous iteration's readers before the DMA
                    # overwrites what they were reading. One tile in flight, so
                    # there is no second buffer to hide behind.
                    _s_barrier()
                    kv_gmem_to_lds.stage_k(tile_start)
                    kv_gmem_to_lds.stage_v(tile_start)
                    if const_expr(LDS_BIAS):
                        kv_gmem_to_lds.stage_bias(j)
                    _s_waitcnt(0)
                    _sched_barrier(0)
                    _s_barrier()  # every wave's DMA has landed, not just mine

                    # -- GEMM1. S, raw. The scale and the LSE subtract follow.
                    v_s = gemm_helper.qk(k_reader.load_k(0), q_all_bf16)
                    v_s = softmax_helper.scale_and_sub_lse(
                        v_s, ctx.c_sm_scale_log2e, lse2
                    )
                    # **This also adds the bias**, on the 32-row family:
                    # `ParitySoftmaxHelper.seq_pad_mask_if_needed` folds
                    # `_add_bias_inplace` in ahead of the tail mask, which is
                    # exactly the order wanted -- `bias * log2e` onto the scaled
                    # score, then `-inf` for the columns that do not exist. Adding
                    # it again here is the B6 `cast_p` mistake in a second costume,
                    # and it is *not* benign: the backward has no softmax
                    # renormalisation, so `P = exp2(S - lse2)` does not absorb a
                    # uniform shift the way the forward's does. Measured at 3.3
                    # relative error, and even a row-*constant* bias failed --
                    # which is what identified it, since shift invariance would
                    # have hidden a double add in the forward.
                    #
                    # Columns past `seqlen_kv` read zero from the buffer bound,
                    # which is a *score of zero*, not an absent key: without the
                    # mask `exp2(0 - lse2)` contributes a spurious P. After the
                    # scale, so the `-inf` it writes never reaches an FMA.
                    #
                    # **Kept under causal too**, where the forward drops it. Under
                    # plain causal it is redundant -- row `i`'s bound is
                    # `i + seqlen_kv - seqlen_q <= seqlen_kv - 1`, so the causal
                    # mask already kills every column past the sequence -- but a
                    # *window* re-points `delta_i32` at an arbitrary right bound
                    # and that argument stops holding. Both guards fire only on the
                    # last tile, so keeping the cheap one unconditionally costs
                    # nothing and removes the case.
                    v_s = softmax_helper.seq_pad_mask_if_needed(v_s, j)
                    if const_expr(traits.CAUSAL):
                        # The forward's mask and the forward's two-sided guard,
                        # inherited whole: `_causal_mask_inplace` applies the right
                        # bound through `delta_i32` and, under `WINDOW`, the left
                        # one. **No transpose read is inside this region** -- it
                        # touches `v_s` only, and `load_v` sits below. CDNA4 11.4
                        # requires EXEC all 1s across `ds_read_b64_tr_b16`, and
                        # B4 is the first phase where a branch exists to violate
                        # it; `test_no_transpose_read_under_a_restricted_exec`
                        # is what keeps that true.
                        v_s = softmax_helper.causal_mask_prologue_if_needed(
                            v_s, j, kv_end_tile=j + fx.Index(1)
                        )

                    # -- GEMM2. dP. V is read through the K path, so this is
                    #    GEMM1's code with the two operands substituted -- and with
                    #    the *other* reader instance, which owns the K-pitch region
                    #    and masks against `hdim_vo`.
                    v_dp = gemm_helper.qk(v_reader.load_k(0), do_all_bf16)

                    # P = exp2(qk_scale*S - lse2). `exp2` is the forward's, split
                    # into two halves there for the pipeline and simply adjacent
                    # here.
                    v_p = softmax_helper.exp2(v_s, 0, ACC_ELEMS)
                    v_p = softmax_helper.exp2(v_p, ACC_ELEMS, ACC_ELEMS)

                    # dS = P * (dP - delta), elementwise over the 32 scores a lane
                    # holds. `delta` is a per-row scalar and a lane owns one row.
                    dp_lo, dp_hi = softmax_helper.v_s_vec_to_lists(v_dp)
                    if const_expr(traits.ENABLE_DROPOUT):
                        # **On dP, before dS.** See `dropout_dp`: the mask belongs
                        # on the gradient of the dropped output, not on `P`, and
                        # `delta` already carries it through the forward's `O`.
                        dp_lo, dp_hi = softmax_helper.dropout_dp(
                            (dp_lo, dp_hi), j, ctx.q_row
                        )
                    ds_lo = [None] * ACC_ELEMS
                    ds_hi = [None] * ACC_ELEMS
                    for r in range_constexpr(ACC_ELEMS):
                        ds_lo[r] = _fmul(
                            v_p[0][r], _fsub(dp_lo[r], delta, ctx.fm_fast), ctx.fm_fast
                        )
                        ds_hi[r] = _fmul(
                            v_p[1][r], _fsub(dp_hi[r], delta, ctx.fm_fast), ctx.fm_fast
                        )

                    # -- GEMM3. dQ += dS . K, with K read transposed out of the V
                    #    slot. `cast_p` gives dS the same K permutation P has, and
                    #    the transpose read is built against that permutation, so
                    #    the two line up with no further shuffle.
                    v_ds = softmax_helper.cast_p((ds_lo, ds_hi))

                    if const_expr(STORE_DB):
                        # **After `cast_p`, before `pv`, and reading the packs.**
                        # `dB = dS`, so nothing here is ordered by *arithmetic* --
                        # the store need only precede `sm_scale`, which is after
                        # the loop. It is ordered entirely by register pressure,
                        # and three placements were measured at `B=2 H=8 S=2048`
                        # (TFLOP/s, on the rungs that spill):
                        #
                        #     variant                        256    384    512
                        #     f32 lists, before cast_p       280    149     56
                        #     packs, after cast_p (this)     315    136     83
                        #     packs, after pv                305    146     73
                        #
                        # Reading the packs is the lever: `dS` exists as 32 f32
                        # *and*, after `cast_p`, as 16 packed bf16, and storing
                        # from the f32 form keeps both alive across the 32-store
                        # sequence. Below head_dim 256 it changes nothing -- the
                        # counts are register-identical, so the allocator was
                        # already sinking the f32 form -- and at 512 it is 1.49x.
                        #
                        # 384 prefers the f32 form by 9%, which the lore says a
                        # sweep cannot settle (it wants interleaved A/B), and it is
                        # an allocator outcome rather than a mechanism: the spill
                        # counts are not monotone with the rate in any of the three
                        # columns. This variant is kept because its one *decisive*
                        # measurement -- 1.49x at 512 -- agrees with the mechanism.
                        db_store.store_tile(v_ds, j, ctx.q_row)

                    v_dq = gemm_helper.pv(v_ds, k_reader.load_v(0), v_dq)

                    loop_results = yield v_dq

                v_dq = _carried(loop_results, traits.D_CHUNKS)
                # `sm_scale` once on the accumulator rather than on every dS: the
                # softmax input is `sm_scale * Q.K^T`, so the factor is linear in
                # the whole sum. AOTriton's `composed_mul_lhs(dq, sm_scale)`.
                softmax_helper.scale_o(v_dq, ctx.c_sm_scale)
                _s_barrier()
                output_store.store_final_o(v_dq, ctx.q_row)

            # head_dim 384 and 512 run a second MFMA family, in its own file: 16
            # rows per wave on `16x16x32`, which halves the loop-invariant Q + dO +
            # dQ that is the whole 512-register file at 32 rows. See
            # `fmha_bwd_dq_m16_gfx950` for why that shape and why `BLOCK_N` 32.
            if const_expr(M16):
                _body = make_m16_dq_body(
                    ctx,
                    traits,
                    kv_gmem_to_lds=kv_gmem_to_lds,
                    db_store=db_store,
                    hdim_qk=hdim_qk,
                    hdim_vo=hdim_vo,
                    hdim_qk_floor=HDIM_QK_FLOOR,
                    hdim_vo_floor=traits.HDIM_VO_FLOOR,
                    store_db=STORE_DB,
                )
            return _body

        def _arm_flags(padded_on, drop_on):
            ctx.PADDED_HEAD = padded_on
            ctx.traits = _arm_traits(drop_on)

        _flags = fx.Int32(rt_flags)
        if const_expr(not runtime_axes):
            _body = _make_arm(PADDED_HEAD, traits.ENABLE_DROPOUT)
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
                def _body():
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
                _body_a = _make_arm(*_fa)
                _body_b = _make_arm(*_fb)

                @flyc.jit
                def _body():
                    if _cond:
                        _arm_flags(*_fb)
                        _body_b()
                    else:
                        _arm_flags(*_fa)
                        _body_a()

        active = ctx.active
        if active is None:
            _body()
        else:

            @flyc.jit
            def _run_body_if_active():
                if active:
                    _body()

            _run_body_if_active()

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def fmha_bwd_dq_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
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
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        rt_flags: fx.Int32,
    ):
        emit_dq_body(
            _WireArgs(
                Q=Q,
                K=K,
                V=V,
                B=B,
                DO=DO,
                DQ=DQ,
                DB=DB,
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
                stride_dq_batch=stride_dq_batch,
                stride_dq_head=stride_dq_head,
                stride_dq_seq=stride_dq_seq,
                stride_b_batch=stride_b_batch,
                stride_b_head=stride_b_head,
                stride_b_seq_q=stride_b_seq_q,
                stride_db_batch=stride_db_batch,
                stride_db_head=stride_db_head,
                stride_db_seq_q=stride_db_seq_q,
                rt_flags=rt_flags,
            )
        )

    @flyc.jit
    def launch_fmha_bwd_dq_gfx950(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
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
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        rt_flags: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Make the build configuration visible to the JIT cache key.
        _ = _cache_tag
        # **Sequences, not batches.** A packed `(1, H, T, D)` call holding N
        # sequences is `batch_size=1, num_seqlens=N`; using the batch extent
        # would launch one program for all N. The forward's expression.
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        num_q_blocks = (fx.Index(max_seqlen_q) + traits.BLOCK_M - 1) // traits.BLOCK_M

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(traits.DAZ)
            else None
        )
        # Head-fastest, the forward's order. Not a free choice there -- it is
        # an L2-locality lever on MI355X's 8 XCDs -- and this kernel streams
        # the same K/V per (batch, head), so it inherits the reasoning.
        fmha_bwd_dq_gfx950_kernel(
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DB,
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
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            stride_db_batch,
            stride_db_head,
            stride_db_seq_q,
            rt_flags,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(num_head_q, num_q_blocks, bs_idx),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _resolve_window_args(window):
        """`(window_left, window_right)` for the wire, as signed i32.

        The forward's, verbatim in behaviour: always a pair, even for a build
        that ignores it, so every build shares one ABI. A *sentinel* rather
        than a bound goes on the wire for the fixed alignments, because the
        kernel resolves it against each sequence's own lengths -- the only
        correct thing to do once varlen means there is more than one pair of
        lengths to resolve against.
        """
        if not traits.CAUSAL:
            if window is not None:
                raise ValueError(
                    "window= requires a causal build; this one has causal=False"
                )
            return 0, 0
        if not traits.WINDOW:
            if window is not None:
                raise ValueError(
                    "this build is not compiled for windows; pass window=True in the builder to get "
                    "generalized sliding-window attention"
                )
            # Bottom-right causal, which is what `delta = seqlen_kv - seqlen_q`
            # already means on this kernel.
            return fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT
        if window is None:
            raise ValueError(
                "a window build requires window=(left, right); pass "
                "(fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT) for bottom-right causal"
            )
        wl, wr = window
        return int(wl), int(wr)

    def _args(
        Q,
        K,
        V,
        DO,
        DQ,
        LSE,
        Delta,
        batch_size,
        seqlen_q,
        seqlen_k=None,
        scale=None,
        db=None,
        bias=None,
        window=None,
        varlen=None,
        num_seqlens=0,
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
            [("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DQ", DQ)],
            q_heads=("DO", "DQ"),
        )
        # Used for the checks and the strides, not the pointers: `prep_tensors`
        # builds those as `fx.Uint8`, which carries alignment 1. `wire_ptr`
        # types each operand from its own tensor instead -- see its docstring.
        del _ptrs
        num_head_q, num_head_k, hdim_qk, hdim_vo = shape_meta

        # **A build without `padded_head` promises the tile *is* the extent**,
        # on both axes, and this is the only place that can be checked. Found
        # by a test of this file's own making: `build(head_dim=128)` handed a V
        # of width 64 resolves to `padded_head=False`, emits no mask at all,
        # and reduces `dP` over the caller's D-axis slack. The answer is finite
        # and 0.70 relative error. A real caller can make the same mistake, so
        # the guard belongs here rather than in the test that found it.
        if not PADDED_HEAD and not (hdim_qk == hdim_vo == BLOCK_DMODEL):
            raise ValueError(
                f"this build is not compiled for a padded head, so it requires hdim_qk == hdim_vo == "
                f"{BLOCK_DMODEL}; got hdim_qk {hdim_qk}, hdim_vo {hdim_vo}. Pass head_dim_v to the "
                "builder so the D-axis masks are emitted."
            )

        # `abi.varlen_args` is gfx1201's, reused unedited: it encodes the same
        # wire format the forward uses, and it is where the two host-side
        # checks live that no kernel can make -- `batch_size` must be the
        # tensor's batch extent whatever the layout, and a packed
        # `num_seqlens` must agree with the length array. Passing the sequence
        # count where the batch extent belongs launches N programs over a
        # 1-batch tensor and every one of them addresses a plausible row.
        # `varlen=None` is not an error and never was a separate build: it is
        # `bits == 0`, which the decode answers with dense addressing.
        _vl = abi.varlen_args(
            bool(knobs.strides_constexpr),
            varlen,
            seqlen_q,
            seqlen_k,
            Q,
            batch_size,
            num_seqlens,
        )

        # **`(batch * heads, tokens)`, and the shape is shared with dK/dV.**
        # Both backward kernels take the same two row tensors and read them
        # with the same `fmha.lse_row_addressing`, so the host check is the
        # shared `abi.row_tensor_arg` rather than a second spelling of it.
        # This is the layout the forward writes LSE in -- `q_head_idx *
        # seq_len_v + q_row` inside a per-batch slab -- viewed at rank 2; a
        # caller holding `(B, H, S)` passes `lse.view(-1, S)`, which is free
        # on a contiguous tensor.
        #
        # **Varlen moves the token pitch**, and this is the one input where
        # that is invisible from the tensor alone: the kernel derives the
        # pitch from `lse_token_pitch` -- the batch total for a stacked Q side,
        # `max_seqlen_q` otherwise -- rather than reading a stride, so a
        # caller who sized LSE by `max_seqlen_q` under a packed layout gets a
        # plausible value for the wrong row. `row_tensor_arg` checks the
        # declared layout against the bits, which is the only place that can
        # be done. The forward's P4 got this wrong in a way only `_TH`
        # exposed.
        rows_expected = int(num_seqlens or batch_size) * num_head_q
        for name, t in (("logsumexp", LSE), ("delta", Delta)):
            if t is None:
                raise ValueError(
                    f"{name} is required: the backward reads it, it is not recomputed"
                )
            abi.row_tensor_arg(t, name, num_head_q, seqlen_q, varlen)
            if varlen is None and t.shape[0] != rows_expected:
                raise ValueError(
                    f"{name} must be ({rows_expected}, {seqlen_q}); got {tuple(t.shape)}"
                )

        # A dB build must be handed a tensor and a build without dB must not
        # be: silently ignoring one returns gradients that are the right shape
        # with the bias gradient missing, and it is only ever passed by a
        # caller who believes it is being written.
        if STORE_DB and db is None:
            raise ValueError(
                "this build has store_db=True and requires a (batch, num_heads_q, seqlen_q, seqlen_k) dB"
            )
        if db is not None and not STORE_DB:
            raise ValueError(
                "this build was not compiled for dB; pass store_db=True to the knobs"
            )
        if db is not None:
            # **dB follows Q's batch and row layout, not a dense one.** Its
            # descriptor is rebased at the same `(batch, head, row origin)` dQ
            # uses, so under a packed Q the rows are packed too and the batch
            # extent is Q's. Only the KV axis is sized independently, to
            # `max_seqlen_k`, because a row's live columns are a per-sequence
            # prefix of it.
            want = (int(Q.shape[0]), num_head_q, int(Q.shape[2]), int(seqlen_k))
            if tuple(db.shape) != want:
                raise ValueError(
                    f"dB must follow Q's batch and row layout with a seqlen_k column axis, {want}; "
                    f"got {tuple(db.shape)}"
                )
            if db.stride(3) != 1:
                raise ValueError(
                    f"dB needs a contiguous seqlen_k axis; strides are {tuple(db.stride())}"
                )
            if db.dtype != Q.dtype:
                raise ValueError(f"dB must match Q's dtype ({Q.dtype}), got {db.dtype}")
        # **The seed and offset the forward *reported*, not what its caller
        # passed.** Under a captured graph the caller's offset is a pointer the
        # forward re-reads, so the two need not be equal -- and the mask has to
        # be regenerated bit-identically or the gradients are quietly wrong.
        # `abi.dropout_args` is gfx1201's: it turns the probability into the
        # i32 threshold the raw random is compared against and the `1/(1-p)`
        # survivor scale, once per call, and keeps the counter as the
        # (pointer, immediate) pair torch splits it into.
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
                "this build was not compiled for dropout; pass dropout=True to the builder"
            )

        # **The bias *input* is a different path from the dB *output***, and
        # having written dB is the part that gets mistaken for done. `B` is
        # added to the scores before the softmax; `dB` is the gradient with
        # respect to it. A build can read one, write the other, both or
        # neither.
        if traits.BIAS_TYPE and bias is None:
            raise ValueError(
                "this build has bias=True and requires a (batch, num_heads_q, seqlen_q, seqlen_k) tensor"
            )
        if bias is not None and not traits.BIAS_TYPE:
            # Silently ignoring it returns dense attention's gradient: right
            # shape, finite, wrong. It is only ever passed by a caller who
            # believes it is being applied.
            raise ValueError(
                "this build was not compiled for bias; pass bias=True to the builder"
            )
        if bias is not None:
            want = (int(Q.shape[0]), num_head_q, int(Q.shape[2]), int(seqlen_k))
            if tuple(bias.shape) != want:
                raise ValueError(
                    f"bias must follow Q's batch and row layout, {want}; got {tuple(bias.shape)}"
                )
            if bias.stride(3) != 1:
                raise ValueError(
                    f"bias needs a contiguous seqlen_k axis; strides are {tuple(bias.stride())}"
                )
            if traits.BIAS_STAGE == "lds" and not bias_chunk_contract_ok(
                bias.data_ptr(), bias.stride()[:3], seqlen_k
            ):
                raise ValueError(
                    "an LDS-staged bias reads in 16-byte chunks: the pointer must be 16-byte "
                    f"aligned, the batch/head/row strides multiples of 8 elements, and a non-zero row stride at least "
                    f"ceil8(seqlen_k)={-(-int(seqlen_k) // 8) * 8}; got ptr%16={bias.data_ptr() % 16}, strides {tuple(bias.stride())}"
                )
        bias_t = bias if bias is not None else DQ
        # `_seq_q` is the *query* axis, matching `stride_db_seq_q` two lines
        # down and the 24 other occurrences the rename settled on. `_seq`
        # alone does not say which sequence; `_seq0` would reintroduce the
        # numeric slots the stride rename removed, and leaves `_seq_k` with
        # nothing to be called if the k axis ever stops being contiguous.
        bias_st = (
            tuple(int(x) for x in bias.stride()[:3]) if bias is not None else (0, 0, 0)
        )

        db_t = db if db is not None else DQ
        db_st = tuple(int(x) for x in db.stride()[:3]) if db is not None else (0, 0, 0)

        return (
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
            wire_ptr(DQ),
            wire_ptr(db_t),
            wire_ptr(LSE, fx.Float32),
            wire_ptr(Delta, fx.Float32),
            int(batch_size),
            _vl[1],
            _vl[2],
            _vl[3],
            _vl[4],
            _vl[0],  # varlen_bits
            int(num_seqlens),
            *_vl[5:],  # max_seqlen_q, max_seqlen_k -- the decode's MAX fallback
            *_resolve_window_args(window),
            _dp[0],
            _dp[1],
            _dp[2],
            _dp[3],  # idropout_p
            _dp[4],  # dropout_scale
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
            *db_st,
            int(not (hdim_qk == hdim_vo == BLOCK_DMODEL))
            | (2 if dropout_p is not None else 0),
        ), stream

    def _launch(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return abi.run_compiled(
                _COMPILED,
                launch_fmha_bwd_dq_gfx950,
                *packed,
                stream if stream is not None else fx.Stream(None),
            )

    def _compile(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return flyc.compile(launch_fmha_bwd_dq_gfx950, *packed, fx.Stream(stream))

    _launch.compile = _compile
    _launch.traits = traits
    _launch.knobs = knobs
    _launch.emit = emit_dq_body
    _launch.SharedStorage = SharedStorage
    _launch.cache_tag = _cache_tag
    return _launch


def build_fmha_bwd_dq_gfx950_module(arch="gfx950", **kwargs):
    """Keyword front end: name a problem, get the policy's schedule.

    `causal` defaults to **False** here where `FmhaInputMetadata` defaults it
    to True. B2 implements only the dense case and `build_traits` refuses the
    other, so inheriting the forward's default would make every unqualified
    call raise -- and a caller who wants causal should be told it is B4, not
    told to pass an argument that is then rejected.
    """
    from dataclasses import fields as _fields

    meta_fields = {f.name for f in _fields(FmhaInputMetadata)}
    meta_kwargs = {"causal": False}
    meta_kwargs.update({k: v for k, v in kwargs.items() if k in meta_fields})
    meta = FmhaInputMetadata(**meta_kwargs)
    knob_kwargs = {k: v for k, v in kwargs.items() if k not in meta_fields}
    knobs = bwd_dq_knobs(arch, **knob_kwargs)
    if not isinstance(
        knobs, BwdDqKnobs
    ):  # pragma: no cover -- defensive, the factory is typed
        raise TypeError(f"expected BwdDqKnobs, got {type(knobs).__name__}")
    return build_fmha_bwd_dq_gfx950_module_primary(meta, knobs.resolve(meta))
