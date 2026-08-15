# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense flash-attention prefill kernel for gfx942 (CDNA3).

Port of the gfx950 dense prefill kernel (``kernels/gfx950/attention_dense.py``,
from PR #9480) to CDNA3.

This is a SEPARATE kernel from the gfx950 sibling by design: the gfx950 body bakes
in CDNA4-only primitives that do not exist on gfx942, so the two algorithms
genuinely diverge (per ``dsl_docs/architecture/multi_arch_data_layout.md``: split
into per-gfx modules the moment the K-loop shape / memory path / fused phases
differ). Keeping them in separate files also keeps the gfx950 golden IR
(``tests/golden/attention_dense_ir_sha256.json``) byte-identical by construction.

Why this is a port, not a copy (the CDNA3 deltas)
-------------------------------------------------
  * **MFMA atom.** gfx950 uses the wide-K ``mfma_f32_32x32x16`` atom (K=16). CDNA3
    has NO 32x32x16 fp16/bf16 instruction — only ``mfma_f32_32x32x8`` (K=8, the
    ``.1k`` bf16 variant). The 32x32x16 warp tile lowers to 2x ``32x32x8`` on
    gfx942 (K-loop doubling; A/B repack ``<8xelem>`` -> ``<4xelem>``/lane). The
    C-output lane layout is IDENTICAL between the two atoms, so softmax reductions
    and the epilogue are unchanged. See ``helpers.attention.mfma_32x32x8_for_dtype``.
  * **Conflict-free V.** gfx950's CK-1 transposed PV uses ``ds_read_b64_tr_b16``
    (transpose read) for the half-local V load. gfx942 has NO ``ds_read_tr16`` and
    NO ``permlane32_swap``. The CDNA3 vehicle is a ``perm_b32`` STORE-path transpose
    (CK ``transpose_vectors`` masks) into a transposed ``V_lds`` — the same vehicle in
    ``kernels/gfx942/attention_tiled_2d.py`` (``use_conflict_free_v_store``). Its
    parked "1/8 sign-flip" symptom was reproduced-or-disproven before anything was
    lifted: DISPROVEN cross-part (both the 228- and 304-CU parts, bit-identical, no
    sign mismatches), so the vehicle is proven correct rather than merely untested.
    It is lifted here for D128 fp16 as :func:`_use_cfvst` — the barrier caveat is
    that V publication becomes an in-loop ``ds_write``, so the tile-START rendezvous
    needs an lgkmcnt drain (``sync_lds_only``), not a bare barrier. The disproof
    procedure and its evidence are recorded in the gfx942 dense-prefill
    optimization plan, which is kept outside the repo.
  * **LDS / occupancy.** gfx942 has 64 KB LDS/CU (vs gfx950's 160 KB); occupancy /
    ``num_persistent`` / block sizing must be re-derived for the 228- and 304-CU
    gfx942 parts rather than inherited from the gfx950 tuning.

Problem category (drives the optimization order)
------------------------------------------------
**Measured bound (rocprofv3 PMC over the Step-0 config sweep): occupancy-bound ->
MFMA-starved.** MFMA issue rate sits far under the runbook's compute-bound
threshold while waves/CU sit at a small fraction of the per-CU maximum; L2 hit rate
and the memory-stall counter rule out HBM/L2 as the primary limit, and no in-scope
config spills. Per the runbook §3.1a decision tree -- a short-circuiting chain whose
FIRST branch is ``occupancy < 0.5`` -- the classification resolves at occupancy and
never reaches the ``LDS_BOUND`` branch.

**This SUPERSEDES the earlier "LDS-bank-conflict-bound" framing** that was inherited
from the gfx942 tiled-2D kernel and used to order P0-P5. That framing was a good
guide and it paid off -- P1 conflict-free V and the P3 D64 K bank-pad each removed a
large bank-conflict cost -- but it does not describe the kernel that remains. Two
findings retire it:

  * Post-P1/P3 the residual bank-conflict RATE no longer tracks throughput: within a
    fixed shape/dtype, configs with several times the conflict rate are level with or
    faster than their low-rate siblings. The counter is conflicts per LDS-active
    cycle, so a lever that removes LDS instructions (cfvst: 128 -> ~32 LDS instrs per
    tile at D128) shrinks the denominator and can RAISE the rate while lowering the
    cost. Rate is not cost; only an ablation prices LDS.
  * Occupancy trips the decision tree first, so LDS conflicts cannot be the governing
    limiter regardless of their rate.

What still follows from this, unchanged: NOT compute-bound and NOT HBM-bound, so
compute-side scheduling levers that win on the compute-bound gfx950 dense kernel
(``s_setprio``, diagonal two-phase peel) stay PROVEN-NEGATIVE on gfx942 and are NOT
ported, and bandwidth-saving levers stay demoted. What changes: the dominant
remaining lever is **occupancy** (D128 -> 2 WG/CU), not a further LDS-conflict fix.
That is a combined-constraint problem -- it needs LDS <= 32 KB AND the total-register
floor cut TOGETHER (D128 carries a large AGPR residency on top of its architected
VGPRs). Dropping the LDS pads alone to buy the LDS half is PROVEN-NEGATIVE and
catastrophically so: occupancy does not move (the register floor co-limits) and the
K/V bank conflicts come straight back.

Measured conflict rates, pipe utilisation and per-lever deltas are recorded outside
the repo -- see the optimization plan and the protected results page.

Implementation status (the optimization plan holds the full ordered work list)
-----------------------------------------------------------------------------
  * P0  enablement + 32x32x8 atom + K-loop doubling ............ DONE (this file)
  * P1  conflict-free V (perm_b32 store-path transpose) ........ DONE (D128 fp16)
  * P2  exp2_fast + fused/lazy rescale ........................ DONE (exp2_fast all
        but bf16 D128; fused rescale bit-identical, enables bf16 D64 exp2_fast)
  * P3  occupancy: waves-per-eu tune (bf16 D64 -> wpe4 = 2 WG/CU),
        D64 K bank-pad, wide4 (WG=256), K single-buffer ........... IN PROGRESS
        (waves-per-eu DONE via _tuned_waves_per_eu; **D64 K bank-pad DONE and ADOPTED**
        -- cross-part-confirmed large win on both D64 dtypes, wired per-config via
        the SHARED ``spec.lds_k_group_pad`` (the same field and formula gfx950
        uses -- one lever, not a per-arch duplicate), golden re-blessed against the
        shipped path; wide4 / K-single-buf still TODO. D128 -> 2 WG/CU is the open
        item and needs the LDS and register floors cut together -- see "Problem
        category" above. ``block_m`` is a further occupancy axis: contrary to the
        gfx950 sibling's "kernel FAULTS at other values" comment, every ``BLOCK_M``
        use in THIS body is parametric (the invariant is ``BLOCK_M == 32 * WAVES``),
        but shrinking it adds CTAs without adding a CTA/CU -- LDS is BLOCK_M-
        invariant -- and costs VGPRs on the K-side DMA addressing, so it is a modest
        win on two of four shapes and a loss elsewhere; now exposed as
        ``Gfx942DenseTuning.block_m`` so re-testing it is a sweep argument rather
        than a source edit, and still DEFAULT-OFF at the shared ``_BLOCK_M``.)
  * P4  persistent grid-stride + qb/hkv_major decode .......... DONE (both decodes;
        shared _run_work_item body; auto-on for large Sq via dispatch, validated
        cross-part for every config by the ON/OFF ablation -- KEEP everywhere)
  * P5  diagonal masking (re-test only), partial-vmcnt prefetch  N/A (see below)

P5 is deliberately no-op for this kernel (a decision, not code):
  * Diagonal two-phase peel: the plan gates it on "re-test ONLY if the bound shifts
    to compute after P1-P4." It has not -- the rocprof sweep confirms D128 is still
    occupancy/latency-bound at 1 WG/CU and nowhere near compute-bound -- and the peel
    was proven negative on gfx942 tiled_2d via register pressure, which D128 (at its
    register floor) is the most exposed to. Keep the single-phase mask.
  * partial-vmcnt software prefetch: N/A here, but NOT for the reason previously
    recorded. It is a double-buffered pipelining lever and this kernel is
    single-buffered (NBUF=1), so there is no prefetch to partially overlap. The old
    "NBUF=2 does not fit 64 KB LDS at D128" justification is only true at the shipped
    ``block_n=64``: at ``block_n=32`` D128 LDS roughly halves and NBUF=2 DOES fit.
    That door is closed by measurement instead -- ``block_n=32`` was GPU-timed in the
    Step-0 funnel and is proven-negative for fp16 D128 and only part-dependently
    positive for bf16 D128 (halving the KV tile doubles the tile/grid count and the
    extra loop/barrier overhead outweighs the LDS relief while occupancy stays
    pinned), so it is not wired and the prefetch it would unlock is unreachable on
    the shipped config. Independently, bounding the live set with partial ``vmcnt=N``
    waits entangles with the K async DMA, which shares the same counter -- see the
    reverted cfvst-chunking experiment recorded in the ``V_ITEMS`` NOTE below.

P1 conflict-free V (:func:`_use_cfvst`): V is stored TRANSPOSED via a perm_b32
store-path transpose so the PV A-operand read is a contiguous ds_read_b64 instead of
P0's 4 element-wise ds_read_u16 (128 -> ~32 LDS instrs/tile at D128). Gated to
**D128 fp16**: D64 is VGPR-bound so the register round-trip regresses it, and D128
bf16 spills over the waves-per-eu=2 cap (a P2/P3 register-headroom item). The lever
is numerically identity-preserving (same MFMA operands; only the LDS layout and read
width change) -- the cohort's ``max_abs`` is unchanged from the naive path. The
conflict-free vehicle was proven correct cross-part before being lifted (see the
optimization plan for the disproof procedure).

P0 is CORRECTNESS-FIRST: the naive-V path (D64 / bf16-D128) is non-pipelined (a
single LDS buffer) and reads V element-wise; the remaining perf levers (P2-P4) layer
on top. It is validated against an fp32 SDPA reference across the in-scope cohort.

:func:`supports_attention_dense` is the SINGLE gate: it rejects every spec
:func:`build_attention_dense` cannot emit -- including the modes deferred to later
phases -- so ``supports_attention_dense(spec)[0] is True`` implies the build
succeeds. Dispatch keys on it, which is what stops an out-of-scope request from
selecting this arm and then failing at build time.

The compile-time spec (:class:`AttentionDenseSpec`) is REUSED from the gfx950
module: batch / seqlen / heads / head_size / causal / dtype / knobs are arch-neutral
(compile-time shape + tuning). gfx942-specific tuning DEFAULTS (e.g. num_persistent
for the gfx942 CU count) are applied in the builder / dispatch layer, not by forking
the dataclass.

Knobs that are gfx942-ONLY -- because gfx950 either cannot emit them or emits them
from a different derivation -- live in :class:`Gfx942DenseTuning` instead, a
gfx942-owned struct threaded through every entry point with a shipped default. It
never touches ``AttentionDenseSpec.kernel_name()`` (the emitted symbol, and a gfx950
golden), and it is invisible to dispatch: production always takes the default.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import (
    IRBuilder,
    KernelDef,
    PtrType,
    VectorType,
    BF16,
    F16,
    F32,
    I32,
    I64,
)
from rocke.helpers.attention import mfma_32x32x8_for_dtype

# The spec is arch-neutral (compile-time shape + tuning knobs); reuse it rather
# than fork the dataclass. gfx942-specific defaults live in the builder/dispatch.
from kernels.gfx950.attention_dense import (
    AttentionDenseSpec,
    _BLOCK_M,
)

# C-output lane maps: IDENTICAL between the 32x32x8 (gfx942) and 32x32x16 (gfx950)
# atoms (mfma_atom_catalog.md), so the softmax reductions + epilogue port verbatim.
from kernels.gfx942.attention_tiled_2d import _mfma_32x32_c_row, _mfma_32x32_c_col

LOG2E = 1.4426950408889634
_DTYPE_IR = {"bf16": BF16, "fp16": F16}

# Pipeline constants (mirror gfx950; this body is NBUF=1, a single LDS buffer).
# The kernel body tiles on the SHARED _BLOCK_M (256 query rows per CTA = 8 wave64s),
# attention_dense_grid sizes the launch grid from it, and supports_attention_dense's
# block_n divisibility check uses it -- one constant, imported from the gfx950 sibling,
# so the grid and the kernel body cannot disagree silently (rows written twice/never).
#
# What that single constant still has to satisfy for THIS body: the wave count is
# WAVES = _BLOCK_M // 32, a FLOOR. At a non-multiple of 32 the emitted wave count
# under-covers the query tile and the top _BLOCK_M % 32 rows of every block get no
# wave at all -- they are never computed and never written, a silent wrong answer with
# no fault. Bind the invariant at import instead of by comment. Not an `assert`:
# `python -O` strips asserts and this guards a silent wrong answer.
if _BLOCK_M % 32 != 0:
    raise ValueError(
        f"_BLOCK_M ({_BLOCK_M}) must be a multiple of 32: the gfx942 dense body "
        f"derives its wave count as WAVES = _BLOCK_M // 32, which floors, so a "
        f"non-multiple leaves the top {_BLOCK_M % 32} query rows of every block "
        f"uncovered by any wave -- never computed and never written"
    )


@dataclass(frozen=True)
class Gfx942DenseTuning:
    """gfx942-private codegen knobs for the dense prefill body.

    SEPARATE from :class:`AttentionDenseSpec` on purpose, and not a fork of it. The
    spec is the arch-neutral compile-time PROBLEM (shape, plus the tuning gfx950 also
    owns) and it is emitted into the IR as the kernel symbol via
    ``spec.kernel_name()`` -- extending it would move an 18-case gfx950 golden
    (``platform/tests/golden/rocke_representative_ir_sha256.json``) and the C++/Python
    parity gate. This struct is the gfx942 SWEEP surface instead: dispatch leaves
    every field at its default, so it is invisible in production and fully
    addressable from a harness.

    Tri-state fields: ``None`` means "use the measured policy"
    ---------------------------------------------------------
    ``use_cfvst`` / ``use_exp2_fast`` / ``waves_per_eu`` default to ``None``, which
    resolves through the shipping policy (:func:`_use_cfvst`, :func:`_use_exp2_fast`,
    and ``spec.waves_per_eu`` -- which the gfx942 dispatch fills from
    :func:`_tuned_waves_per_eu`). That tri-state is the whole point of the struct: a
    harness that omits a VALUED field freezes the config at whatever the default
    happened to be the day it was written and then silently reports a stale verdict
    (that is exactly how a real +79% got reported as -17% in this tree). A harness
    that omits a ``None`` field auto-tracks whatever ships. Omission becomes safe.

    The policy functions stay the SINGLE resolution point -- their docstrings are the
    evidence ledger, and their verdicts are deliberately NOT copied into the defaults
    here, so there is nothing to drift.

    Naming / launcher-cache safety
    ------------------------------
    Every field that can change codegen is tagged into :func:`gfx942_kernel_name`,
    but ONLY when it differs from the default (or, for the tri-state fields, from
    what the policy would have produced) -- so at shipped defaults the names, and the
    goldens, are byte-identical. Conditional tagging is the existing idiom
    (``AttentionDenseSpec.kernel_name`` emits ``kpad{N}`` only on the packed path and
    ``persist{N}`` only when persistent), and it is load-bearing rather than
    cosmetic: ``_DENSE_LAUNCHER_CACHE`` is keyed on the kernel name and its
    ``assert art.kernel_name == key`` PASSES on a name collision, so an untagged
    IR-affecting knob silently serves a stale binary. This kernel has shipped that
    exact bug twice (``batch``, then ``waves_per_eu``).

    :func:`supports_attention_dense` validates this struct, not just the builder: the
    module contract is that ``supports(spec, tuning=t)[0] is True`` implies
    ``build_attention_dense(spec, tuning=t)`` succeeds.
    """

    # block_m: query rows per CTA. The wave count is block_m // 32 and the CTA is
    #   waves*64 threads. Defaults to the IMPORTED ``_BLOCK_M`` rather than a repeated
    #   literal so the default cannot drift from the constant the grid helper and the
    #   gfx950 sibling pin. It is a gfx942-only tunable, and must stay one: the gfx950
    #   builder FAULTS at any other value (its causal mask and P relayout hardcode
    #   256 -- ``kernels/gfx950/attention_dense.py:83-84``) and pins the module
    #   constant regardless of any field, so this can never become a shared-spec
    #   field. Every ``block_m`` use in THIS body is parametric.
    #   Status: an OPEN occupancy axis. Shrinking it adds CTAs without adding a
    #   CTA/CU (the LDS footprint is block_m-invariant) and costs VGPRs on the K-side
    #   DMA addressing, so it measured a modest win on two of four shapes and a loss
    #   elsewhere -- which is why it is a sweep knob here instead of a source edit.
    block_m: int = _BLOCK_M

    # lds_row_pad: K_lds per-ROW bank-conflict pad, in elements. Applied only when one
    #   K row is packed per async-DMA instruction (D128 here); see
    #   :func:`_lds_row_stride` for why D64 cannot carry a per-row pad and uses the
    #   per-GROUP ``spec.lds_k_group_pad`` instead. The two are DIFFERENT,
    #   mutually-exclusive levers that merely happen to share the value 8.
    #
    #   SETTLED -- but scoped strictly to REMOVAL. Dropping this pad (and
    #   ``v_row_pad``) to cut D128 LDS under the 32 KB a 2nd WG/CU needs is
    #   PROVEN-NEGATIVE and catastrophically so: D128 stays at 1 WG/CU even unpadded
    #   because the register floor co-limits, so the pads come out for nothing and the
    #   K/QK (and cfvst V) bank conflicts come straight back. Do not re-attempt pad
    #   REMOVAL.
    #
    #   The VALUE is a different question and it is OPEN. 8 is INHERITED from the
    #   gfx950 sibling and has never been re-derived for gfx942. Two concrete reasons
    #   to distrust it:
    #     * Bank geometry does not transfer. gfx950 has 64 LDS banks, gfx942 has 32
    #       (``platform/dsl_docs/optimization/arch/gfx950.md`` §21.2 "LDS specifics"
    #       table; ``platform/dsl_docs/optimization/arch/gfx942.md`` §21.2), and the
    #       shared pad derivation is written against the 64-bank model verbatim --
    #       "LDS delivers 64 banks x 4 B = 256 B/cycle" (``AttentionDenseSpec``'s
    #       ``lds_k_group_pad`` comment in ``kernels/gfx950/attention_dense.py``). The
    #       same gfx950 table also lists the preferred swizzle as *padding* for
    #       gfx950 and *XOR* for gfx942, so even the FORM of the fix is not obviously
    #       transferable, let alone the magnitude.
    #     * 8 did not survive a sweep on the arch it came from. gfx950's own sweep of
    #       the analogous V pad measured conflicts {pad 0: 30, pad 8: 29, pad 16: 11,
    #       pad 32: 0} -- i.e. +8 was essentially indistinguishable from no pad at all
    #       (``kernels/gfx950/attention_dense.py`` module docstring / ``_LDS_PAD_V``).
    #   Re-deriving both pad VALUES on gfx942 is the pad-value sweep tracked in the
    #   optimization plan; THIS field is the knob that sweep turns.
    lds_row_pad: int = 8

    # v_row_pad: V^T row (token axis) bank-conflict pad, in elements, for the P1
    #   conflict-free-V store. V_lds is transposed to [D, block_n] (dim-major, token
    #   inner) so the PV A-operand read is a contiguous ds_read_b64; the pad spaces
    #   the dim rows so the per-lane 4-token reads land in distinct banks (token
    #   stride block_n+8 dwords -> 8-bank / 4-way at block_n=64, matching the D128 QK
    #   path). Lifted from the proven ``attention_tiled_2d`` cfvst vehicle (v_pad=8).
    #   Same status as ``lds_row_pad``: pad REMOVAL is settled-negative, the pad VALUE
    #   is inherited-and-not-re-derived-for-gfx942 and therefore OPEN -- and this is
    #   the very pad whose gfx950 sweep found 8 ~= no pad at all (see above).
    #   Resolved via :meth:`resolved_v_row_pad` (``None`` -> :func:`_v_row_pad`): 64 on
    #   the cfvst path (fp16-D128) to widen V_lds to pow2 128 so the swizzle engages,
    #   else 8. Every read goes through that one resolver, so the budget cannot drift
    #   from the allocation.
    v_row_pad: int | None = None

    # use_cfvst: force the P1 conflict-free perm_b32 store-path V transpose on/off.
    #   None (default) -> :func:`_use_cfvst`, which owns the measured verdict.
    #   Forcing it ON where the policy says OFF is REJECTED by
    #   supports_attention_dense (D64 regresses under it -- VGPR-bound; bf16 D128
    #   spills past the waves-per-eu cap), so this field can only turn cfvst off, or
    #   pin it on where it already is.
    use_cfvst: bool | None = None

    # use_exp2_fast: force the P2 single-instruction exp2 on/off.
    #   None (default) -> :func:`_use_exp2_fast`, which owns the measured verdict
    #   (on everywhere except bf16 D128, which spills). Numerically safe in both
    #   directions here -- both softmax arguments are always <= 0 -- so unlike
    #   ``use_cfvst`` this one is a pure perf A/B and is not gated.
    use_exp2_fast: bool | None = None

    # waves_per_eu: override the emitted ``amdgpu-waves-per-eu`` attribute.
    #   None (default) -> ``spec.waves_per_eu``, which the gfx942 dispatch spec
    #   factory fills from :func:`_tuned_waves_per_eu` -- so ``None`` auto-tracks the
    #   measured policy end to end without this struct restating it. The RESOLVED
    #   value is what lands in the ``wpe{N}`` name token and in the kernel attribute,
    #   so the name and the binary cannot disagree.
    waves_per_eu: int | None = None

    # iglp: emit runbook lever 7 (optimization_runbook.md §8.4) -- ``iglp_opt``, the
    #   canned backend MFMA / ds_read / ds_write interleave (llvm.amdgcn.iglp.opt),
    #   distinct from the proven-negative s_setprio / sched_barrier hints (plan §2
    #   DO-NOT-PORT). Its precondition is in-loop ds_write traffic to interleave
    #   against the MFMAs -- which only the P1 cfvst path has (V is stored to V_lds
    #   inside the loop; the naive path is direct-load, 0 ds_write, so iglp is a
    #   priori neutral). Placed once at the top of the main-loop body per the runbook.
    #
    #   RESOLVED -- do not re-attempt. This is no longer a pending probe: the GPU
    #   timing it was gated on has been run, cross-part, on the cfvst config it was
    #   supposed to help.
    #     * Static (runbook §8.4): the intrinsic is a compile-time scheduler DIRECTIVE
    #       and leaves NO runtime instruction -- the ISA opcode histogram is unchanged
    #       and ds_read / ds_write / mfma counts are identical, with no spill or LDS
    #       delta. So the effect could never be confirmed by artifact and had to be
    #       timed.
    #     * Timed: PROVEN-NEUTRAL on both parts. Its only visible effect is shifting
    #       the architected-VGPR / AGPR split (88/128 -> 128/128 at fp16 D128) at
    #       equal total registers, which does not move the clock.
    #   The neutrality is itself the useful result, per runbook §8.6 (barrier-bound vs
    #   schedule-bound): a canned MFMA/DS interleave buying nothing means the main
    #   loop is NOT schedule-bound -- it is barrier-rendezvous-bound, which no
    #   scheduling hint can fix. That matches the measured occupancy bound in the
    #   module docstring, and it is why the whole scheduling-hint family stays
    #   unported. Stays default OFF; it is kept as a knob only because it toggles IR.
    iglp: bool = False

    def resolved_use_cfvst(self, spec: AttentionDenseSpec) -> bool:
        """Resolved conflict-free-V decision (``None`` -> :func:`_use_cfvst`)."""
        if self.use_cfvst is None:
            return _use_cfvst(spec.head_size, spec.dtype)
        return bool(self.use_cfvst)

    def resolved_v_row_pad(self, spec: AttentionDenseSpec) -> int:
        """Resolved V^T row pad (``None`` -> :func:`_v_row_pad`): 64 on the cfvst path
        (fp16-D128) so V_lds is a pow2 128 wide and the bank-conflict swizzle engages,
        else 8. Read by the builder's ``V_lds`` alloc AND the ``_lds_bytes`` budget so
        the two cannot drift."""
        if self.v_row_pad is None:
            return _v_row_pad(spec.head_size, spec.dtype)
        return int(self.v_row_pad)

    def resolved_use_exp2_fast(self, spec: AttentionDenseSpec) -> bool:
        """Resolved exp2_fast decision (``None`` -> :func:`_use_exp2_fast`)."""
        if self.use_exp2_fast is None:
            return _use_exp2_fast(spec.head_size, spec.dtype)
        return bool(self.use_exp2_fast)

    def resolved_waves_per_eu(self, spec: AttentionDenseSpec) -> int:
        """Resolved ``amdgpu-waves-per-eu`` (``None`` -> ``spec.waves_per_eu``).

        ``spec.waves_per_eu`` is itself filled from :func:`_tuned_waves_per_eu` by
        ``dispatch.attention.gfx942._dense_spec``, so leaving this ``None`` tracks the
        measured policy rather than pinning today's number -- while a hand-built spec
        keeps whatever it asked for (which is what keeps every existing caller and
        golden byte-identical)."""
        if self.waves_per_eu is None:
            return int(spec.waves_per_eu)
        return int(self.waves_per_eu)


# The shipped configuration. Used both as the default argument of every public entry
# point (so every existing caller keeps working unchanged) and as the comparison
# baseline for the conditional name tags below -- one object, so a default and its
# "is this the default?" test cannot drift apart.
_DEFAULT_TUNING = Gfx942DenseTuning()

__all__ = [
    "AttentionDenseSpec",
    "Gfx942DenseTuning",
    "supports_attention_dense",
    "build_attention_dense",
    "attention_dense_grid",
    "attention_dense_block",
    "attention_dense_signature",
    "gfx942_kernel_name",
    "run_attention_dense_torch",
]


def _tuning_name_tags(spec: AttentionDenseSpec, tuning: "Gfx942DenseTuning") -> str:
    """Name suffix for every :class:`Gfx942DenseTuning` field that is NOT shipped.

    Empty string at the shipped configuration -- which is the property that keeps the
    goldens and every cached kernel name byte-identical -- and non-empty, uniquely, for
    any other tuning. The tri-state fields are compared against what the POLICY would
    have produced, not against the raw ``None``, so ``use_cfvst=True`` on the config
    the policy already turns on is (correctly) the same kernel and the same name.

    Deliberately tagged on "differs from the default", not on "is live in this
    build": at D64 a non-default ``lds_row_pad`` emits identical IR (D64 takes the
    per-GROUP pad instead), so this over-tags there and costs one duplicate compile.
    That direction is free; the other direction -- under-tagging a knob that DOES
    move IR -- silently serves a stale binary out of ``_DENSE_LAUNCHER_CACHE``, which
    is the bug this whole tagging scheme exists to prevent.

    ``waves_per_eu`` needs no tag here: the resolved value is already emitted as the
    ``wpe{N}`` token by :func:`gfx942_kernel_name` itself.
    """
    parts: list[str] = []
    if tuning.block_m != _DEFAULT_TUNING.block_m:
        parts.append(f"bm{tuning.block_m}")
    if tuning.lds_row_pad != _DEFAULT_TUNING.lds_row_pad:
        parts.append(f"krowpad{tuning.lds_row_pad}")
    vp = tuning.resolved_v_row_pad(spec)
    if vp != _v_row_pad(spec.head_size, spec.dtype):
        parts.append(f"vrowpad{vp}")
    cfvst = tuning.resolved_use_cfvst(spec)
    if cfvst != _use_cfvst(spec.head_size, spec.dtype):
        parts.append("cfvst1" if cfvst else "cfvst0")
    e2f = tuning.resolved_use_exp2_fast(spec)
    if e2f != _use_exp2_fast(spec.head_size, spec.dtype):
        parts.append("e2f1" if e2f else "e2f0")
    if tuning.iglp != _DEFAULT_TUNING.iglp:
        parts.append("iglp1" if tuning.iglp else "iglp0")
    return "".join(f"_{p}" for p in parts)


def gfx942_kernel_name(
    spec: AttentionDenseSpec, tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING
) -> str:
    """Kernel name carrying every compile-time-baked parameter the shared name omits.

    ``AttentionDenseSpec.kernel_name()`` omits both ``batch`` and ``waves_per_eu``,
    and this kernel bakes both:

      * ``batch`` sizes the buffer-resource extents, so two specs differing only in
        it collide in a name-keyed cache (launcher / HSACO) and a B>1 launch is
        served the B=1 binary — an out-of-bounds read.
      * ``waves_per_eu`` is emitted as the ``amdgpu-waves-per-eu`` kernel attribute
        and changes register allocation, so the two compile to different binaries.

    Appending both keeps the identity unique. The shared ``kernel_name`` itself
    cannot be extended: it is emitted into the IR as the symbol, so changing it would
    break the gfx950 golden / byte-identity.

    The K row-group pad needs NO tag here: it also changes the emitted K_lds layout
    and do_qk addressing, but it lives in the shared ``lds_k_group_pad`` field, and
    the shared ``kernel_name()`` already emits a ``kpad{N}`` token for it on the
    packed path. Adding a second gfx942 tag would restate the same fact under a
    different name -- exactly the duplication that collapsing the two spec fields
    removed.

    ``tuning`` appends :func:`_tuning_name_tags`, which is EMPTY at the shipped
    configuration: the default name is byte-identical to the pre-tuning-struct one,
    so no golden moves, while any non-default sweep point gets its own identity (and
    therefore its own ``_DENSE_LAUNCHER_CACHE`` entry). The ``wpe`` token carries the
    RESOLVED waves-per-eu, so a ``tuning.waves_per_eu`` override moves the name and
    the emitted attribute together.
    """
    wpe = tuning.resolved_waves_per_eu(spec)
    return (
        f"{spec.kernel_name()}_gfx942_b{spec.batch}_wpe{wpe}"
        f"{_tuning_name_tags(spec, tuning)}"
    )


# In-scope for the gfx942 dense-prefill port. supports_attention_dense rejects
# everything outside this AND every mode the builder cannot emit, so a True result
# from it means build_attention_dense succeeds. That equivalence is what keeps the
# dispatch arm from selecting a spec it cannot build.
_SUPPORTED_DTYPES = ("bf16", "fp16")
_SUPPORTED_HEAD_SIZES = (64, 128)

# 32-bit addressing ceiling. The dense ABI bakes every extent at build time, so the
# limits below are static properties of the spec, not runtime conditions.
_INT32_LIMIT = 2**31


# Elements moved into LDS by ONE async-DMA instruction: 64 lanes x dwords=1 (4 B)
# / 2 B per element. wave64 and a 2-byte dtype are the only cases this kernel emits
# (supports_attention_dense gates dtype to bf16/fp16); an fp8 extension must
# re-derive this or the row stride and the LDS budget go wrong together.
_DMA_ELEMS_PER_INSTR = 64 * 4 // 2  # = 128


def _rows_per_instr(head_size: int) -> int:
    """K/V rows packed into one async-DMA instruction (1 at D128, 2 at D64).

    Single definition of the packing rule that BOTH the LDS row stride and the
    loader's addressing derive from; if those two ever disagree, the padded stride
    and the DMA disagree about row adjacency, which corrupts silently.
    """
    return _DMA_ELEMS_PER_INSTR // head_size


def _use_exp2_fast(head_size: int, dtype: str) -> bool:
    """Whether softmax uses ``exp2_fast`` (one v_exp_f32, no range-reduction guard).

    Enabled everywhere EXCEPT bf16 D128. exp2_fast is a strict VALU reduction and the
    dominant P2 lever on the (post-P1) VALU-bound path, and is always numerically safe
    here -- both softmax args (alpha's m_i - m_new and p's s - m_new) are <= 0, exactly
    exp2_fast's precondition.

    bf16 D128 is the sole holdout. Its ``.1k`` MFMA schedule keeps more registers live,
    and exp2_fast makes the exp result available in one instruction (vs plain exp2's
    ~5-op range reduction), which the scheduler hoists -- lengthening the exp live
    ranges. Measured post-fused-rescale (plan §6.1): bf16 D128 goes 175 VGPR / 0 spill
    (plain exp2) -> 256 VGPR / 22 spill (exp2_fast), over the waves-per-eu=2 cap. The
    fused rescale freed ~28 VGPR but not enough to absorb that hoist on the .1k path;
    fp16 D128 (213, cfvst) and bf16 D64 (215) both have the headroom. A bf16 D128
    exp2_fast unblock is a P3 occupancy/scheduling item. Spill re-verified across the
    fp16/bf16 x D64/D128 cohort.
    """
    return dtype == "fp16" or head_size != 128


def _use_cfvst(head_size: int, dtype: str) -> bool:
    """Whether to feed V through the P1 conflict-free perm_b32 store-path transpose.

    D128 **fp16** only. Measured on gfx942 (both parts):
      * D64 (any dtype): cfvst REGRESSES it -- D64 is VGPR-bound, not LDS-bound
        (plan §6.1), so the register round-trip + perm temps raise pressure without
        relieving the actual bottleneck.
      * D128 bf16: the bf16 ``.1k`` MFMA schedule keeps more registers live, so cfvst
        pushes VGPR over the waves-per-eu=2 cap (256) and SPILLS. fp16 does not
        (it drops slightly, 176 -> 173). The flash x8 ladder is fp16-first anyway
        (tiled_2d keeps bf16 D128 on the narrow path); a bf16 cfvst register-pressure
        fix is a P3 occupancy item.
    Gate on rows-per-DMA == 1 (D128) AND fp16 so the LDS budget and the body agree.
    """
    return _rows_per_instr(head_size) == 1 and dtype == "fp16"


def _v_row_pad(head_size: int, dtype: str) -> int:
    """V^T row (token-axis) pad, in elements. 64 on the cfvst path (fp16-D128) makes the
    ``V_lds`` row width a pow2 128 so the XOR bank-conflict swizzle (``col' = key ^
    ((dim&31)<<2)``) stays in-bounds and engages; 8 (no swizzle, original layout)
    everywhere else, where the transposed-V path is off. The swizzle halves the V-read
    LDS bank conflicts; measured net-positive for fp16-D128 across Sq/GQA with no
    regression, so it rides the same gate as :func:`_use_cfvst`."""
    return 64 if _use_cfvst(head_size, dtype) else 8


def _tuned_waves_per_eu(head_size: int, dtype: str) -> int:
    """Tuned ``amdgpu-waves-per-eu`` per config (P3 occupancy).

    Default 2 (one WG/CU = 2 waves/SIMD at this kernel's ~175-217 VGPR). The one
    override is **bf16 D64 -> 4**, which forces the allocator from 215 VGPR down to
    ~117 (0 spill). D64's LDS is only 16 KB (allows 4 WGs), so the smaller per-wave
    budget lets a SECOND WG co-reside (2 WG/CU), and the bf16 ``.1k`` schedule is
    serialized enough at wpe=2 that its HBM latency is EXPOSED -- the extra resident
    WG hides it. Measured on MI300X: S512 +~77%, S8192 +~48%, S256 neutral, S2048
    ~-1.5% (noise); strongly net-positive, so no seqlen gate.

    Why not the other configs (all measured, all kept at 2):
      * fp16 D64: wpe=3 already reaches 2 WG/CU at 116 VGPR / 0 spill, but its wpe=2
        schedule ALREADY hides the latency via ILP (the 214 VGPR is scheduler headroom
        spent on in-flight LDS reads, plan §6.1) -- halving per-wave registers loses
        more ILP than the 2nd WG buys (S2048 81.5 -> 79.8, slightly negative).
      * D128 (either dtype): LDS-bound at ~35 KB, so 2 x 35 > 64 KB -- no wpe value
        reaches a 2nd WG/CU. Occupancy there is an LDS-footprint problem (P3 K/V-pad
        or single-buffer work), not a waves-per-eu knob.

    Consumed by the gfx942 dispatch spec factory (``_dense_spec``) so the kernel_name
    ``wpe{N}`` tag and the emitted ``amdgpu-waves-per-eu`` attribute always agree.
    """
    if dtype == "bf16" and head_size == 64:
        return 4
    return 2


def _lds_row_stride(
    head_size: int, tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING
) -> int:
    """K_lds / V_lds row stride in ELEMENTS for one head size.

    D128 packs ONE row per async-DMA instruction (64 lanes x 2 elems = 128 elems =
    one D128 row), so the row can carry the bank-conflict pad
    (``tuning.lds_row_pad``). D64 packs TWO rows per instruction, which requires a
    contiguous UNPADDED stride -- a padded row would not be adjacent to the next one,
    so the single instruction could not cover both. D64 therefore pads between
    row-GROUPS instead (``spec.lds_k_group_pad`` / :func:`_k_group_stride`), and
    ``tuning.lds_row_pad`` is inert there.

    Shared by the builder and :func:`supports_attention_dense` so the budget check
    cannot drift from the actual allocation.
    """
    if _rows_per_instr(head_size) == 1:
        return head_size + tuning.lds_row_pad
    return head_size


def _k_group_pad_active(
    spec: AttentionDenseSpec, tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING
) -> bool:
    """Whether the D64 K-LDS row-group bank-conflict pad is EMITTED for ``spec``.

    ADOPTED, not pending. There is NO gfx942-private switch: the pad amount is the
    SHARED ``spec.lds_k_group_pad``, which gfx950 owns and gfx942 reuses verbatim.
    Both arches pad between async-DMA row-GROUPS by the same formula
    (``rows_per_instr * head_size + pad``, see :func:`_k_group_stride`) for the same
    reason, so this is one lever with one field rather than a per-arch duplicate.
    gfx942 derived the pad from a measured ~2x ON/OFF ablation, gfx950 from a
    bank/phase model, and both landed on the same 8-element default.

    Active only on the PACKED path (``rows_per_instr > 1``, i.e. D64 here): D128
    already carries a per-row K pad via :func:`_lds_row_stride`, so it must never
    be re-padded. ``lds_k_group_pad=0`` reproduces the unpadded layout for A/B --
    which is also how the pad's ~2x is priced, no module-level probe constant needed.

    What the pad does
    -----------------
    With a non-zero pad the D64 ``K_lds`` takes a *row-group boundary* pad, so the
    ``do_qk`` K reads drop from a full 32-way bank conflict to 4-way (matching D128)
    WITHOUT losing async-DMA efficiency: the contiguous 2-rows-per-instruction DMA
    still writes one whole group per instruction, and the pad sits at the group
    boundary it never touches. This is the "swizzled LDS layout" arm of the
    hypothesis, and it is strictly better than the other arm that was considered --
    ``ROWS_PER_INSTR=1`` plus half the DMA efficiency.

    MECHANISM (corrected -- a proven negative, do not re-derive)
    -----------------------------------------------------------
    The static codegen A/B is IDENTICAL with the pad on and off: VGPR count, the
    LDS-instruction stream and the ``s_nop`` padding are all unchanged. The
    once-hypothesised "the pad frees registers -> more ILP -> compounding win"
    mechanism is therefore DEBUNKED. The win is purely eliminated bank-conflict
    REPLAY -- the same ``ds_read`` replayed ~8x at 32-way instead of ~4-way -- which
    is a runtime effect and is invisible to any static probe. Re-running a static
    A/B to "explain" this lever will find nothing; only a timed run prices it.

    ``tuning`` is accepted (and ignored) so every LDS-geometry helper takes the same
    ``(spec, tuning)`` call convention: this pad is a SPEC field, not a gfx942 tuning
    field, precisely because gfx950 emits the identical layout from the identical
    formula. Threading the parameter here keeps a future caller from having to know
    which of the two pads it is looking at.
    """
    del tuning  # spec-owned lever; see docstring
    return spec.lds_k_group_pad > 0 and _rows_per_instr(spec.head_size) > 1


def _k_group_stride(
    spec: AttentionDenseSpec, tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING
) -> int:
    """K_lds physical group stride in ELEMENTS when the row-group pad is active.

    A group is ROWS_PER_INSTR (=2 at D64) contiguous rows written by one async-DMA
    instruction; the pad sits at the group boundary (never DMA-touched). Identical
    to the gfx950 builder's ``LDROW = K_GROUP * D + spec.lds_k_group_pad``.
    ``tuning`` is accepted and ignored for the call-convention reason given in
    :func:`_k_group_pad_active`."""
    del tuning  # spec-owned lever; see _k_group_pad_active
    return _rows_per_instr(spec.head_size) * spec.head_size + spec.lds_k_group_pad


def _lds_bytes(
    spec: AttentionDenseSpec, tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING
) -> int:
    """Total LDS footprint: ``K_lds[1, block_n, row_stride] + V_lds[1, D, block_n+pad]``.

    K keeps the natural ``[token, dim]`` layout with :func:`_lds_row_stride` (async
    DMA target). V is TRANSPOSED to ``[dim, token]`` for the P1 conflict-free store, so
    its footprint is ``D * (block_n + tuning.v_row_pad)`` rather than
    ``block_n * row_stride``. 2 bytes/element is exact for every dtype in
    ``_SUPPORTED_DTYPES`` (bf16/fp16) and
    must be revisited if a narrower or wider element type is added. Shared with the
    budget check in :func:`supports_attention_dense` so the two cannot drift.

    When :func:`_k_group_pad_active` (the D64 bank-conflict pad), K instead takes the
    2-row-group layout ``K_lds[1, block_n // rows_per_instr, _k_group_stride]`` --
    +``spec.lds_k_group_pad`` elements per group (D128 and the kpad-off D64 path are
    unchanged).

    ``resolved_v_row_pad`` is read HERE and in the builder's ``V_lds`` allocation; both
    sites go through the same resolver, so the budget cannot silently under-count the
    allocation.
    """
    if _k_group_pad_active(spec, tuning):
        rpi = _rows_per_instr(spec.head_size)
        k_bytes = (spec.block_n // rpi) * _k_group_stride(spec, tuning) * 2
    else:
        k_bytes = spec.block_n * _lds_row_stride(spec.head_size, tuning) * 2
    if tuning.resolved_use_cfvst(spec):
        # V transposed to [dim, token+pad] for the conflict-free store (D128).
        v_bytes = spec.head_size * (spec.block_n + tuning.resolved_v_row_pad(spec)) * 2
    else:
        # V keeps the natural [token, dim] async-DMA layout (D64, naive read).
        v_bytes = spec.block_n * _lds_row_stride(spec.head_size, tuning) * 2
    return k_bytes + v_bytes


def supports_attention_dense(
    spec: AttentionDenseSpec,
    *,
    arch: str = "gfx942",
    tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING,
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for one gfx942 dense-prefill config.

    This is the SINGLE source of truth for what :func:`build_attention_dense` can
    emit: every rejection the builder would make is made here first, so a True
    result implies the build succeeds. Dispatch gates on this, which lets an
    out-of-scope request fall through to another candidate instead of selecting
    this arm and failing at build time.

    That contract covers the ``tuning`` struct too, not just the spec: an illegal
    :class:`Gfx942DenseTuning` is rejected HERE, with a structured reason, rather
    than only blowing up inside the builder (or -- worse -- building a config that is
    known to spill or to read past its tile).

    In scope for this port: gfx942, bf16/fp16, D64/D128, MHA/GQA including
    non-power-of-2 groups, causal or full, the default grid AND the P4 persistent
    grid-stride variant, ``block_n`` dividing the ``tuning.block_m`` query tile,
    within the LDS budget and 32-bit addressing. varlen / ragged / sliding-window are
    later follow-ups (rejected below).
    """
    if arch != "gfx942":
        return False, f"kernels.gfx942.attention_dense is gfx942-only (got {arch})"
    # Type FIRST: every check below dereferences a spec attribute, so a non-spec
    # argument would raise AttributeError out of this (bool, str) API instead of
    # returning the structured rejection the contract promises.
    if not isinstance(spec, AttentionDenseSpec):
        return False, f"spec must be an AttentionDenseSpec, got {type(spec).__name__}"
    if spec.dtype not in _SUPPORTED_DTYPES:
        return (
            False,
            f"gfx942 attention_dense supports {_SUPPORTED_DTYPES}, got {spec.dtype}",
        )
    if spec.head_size not in _SUPPORTED_HEAD_SIZES:
        return False, (
            f"gfx942 attention_dense scope is D{list(_SUPPORTED_HEAD_SIZES)} "
            f"(D256 is served by its own wide-atom candidates), got D{spec.head_size}"
        )
    # Re-run the dataclass validators (shape multiples, GQA divisibility, knob
    # ranges) so a hand-built spec is rejected with a structured reason. Iterate the
    # CLASS fields (a subclass' extra field would be a TypeError here), and catch
    # ZeroDivisionError too: __post_init__ evaluates `seqlen_kv % block_n` BEFORE it
    # validates block_n > 0, so block_n=0 raises ZeroDivisionError, not ValueError,
    # and would escape this (bool, str) API.
    fields = AttentionDenseSpec.__dataclass_fields__  # type: ignore[attr-defined]
    try:
        AttentionDenseSpec(**{f: getattr(spec, f) for f in fields})
    except (ValueError, ZeroDivisionError) as e:
        return False, f"invalid AttentionDenseSpec: {e}"

    # --- Positive extents. Every dataclass validator is a divisibility test, and
    # Python's `%` is sign-following: -256 % 256 == 0 and 8 % -1 == 0, so zero and
    # negative shapes pass all of them. num_query_heads == 0 is the worst -- gqa =
    # Hq // Hkv == 0 emits `sdiv i32 %hq, 0` into the kernel -- and negative extents
    # make the 32-bit checks below vacuously true.
    for _field in (
        "batch",
        "seqlen_q",
        "seqlen_kv",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
    ):
        _value = getattr(spec, _field)
        if _value <= 0:
            return False, f"{_field} must be positive, got {_value}"

    # --- Mode scope. The body implements the default-grid AND the P4 persistent
    # grid-stride variant, both uniform dense self-attention. Checked HERE and not
    # only in the builder so that support() and build() agree on exactly one set of
    # specs. Persistent needs Sq % block_m == 0 (the grid-stride work count
    # W = (Sq // block_m) * Hq * B floors); ragged (the ceil case) is rejected
    # just below, and the seqlen_q check under "Tuning struct" then makes the floor
    # exact for every spec that reaches the persistent builder.
    if spec.varlen:
        return False, "gfx942 attention_dense: varlen not yet supported"
    if spec.ragged:
        return False, "gfx942 attention_dense: ragged not yet supported"
    if spec.sliding_window:
        return False, "gfx942 attention_dense: sliding_window not yet supported"

    # --- Tuning struct (gfx942-private sweep knobs). Validated here rather than only
    # in the builder because the module contract is support() => build(): a knob that
    # only the builder rejects is the same dispatch fall-through hole every other
    # check in this function exists to close. Every branch below is inert at
    # _DEFAULT_TUNING, so the shipped behaviour is unchanged.
    #
    # block_m: the wave count is block_m // 32, a FLOOR. At a non-multiple of 32 the
    # emitted waves under-cover the query tile and the top block_m % 32 rows of every
    # block are never computed and never written -- a silent wrong answer with no
    # fault (the same invariant the module-scope _BLOCK_M guard binds for the
    # default). block_m // 32 * 64 is the CTA thread count, capped at 1024 by the
    # hardware workgroup limit.
    if tuning.block_m <= 0 or tuning.block_m % 32 != 0:
        return False, (
            f"tuning.block_m must be a positive multiple of 32, got "
            f"{tuning.block_m}: the body derives WAVES = block_m // 32, which floors, "
            f"so a non-multiple leaves the top rows of every query block uncovered by "
            f"any wave -- never computed and never written"
        )
    if tuning.block_m // 32 * 64 > 1024:
        return False, (
            f"tuning.block_m={tuning.block_m} needs a "
            f"{tuning.block_m // 32 * 64}-thread CTA, past the 1024-thread workgroup "
            f"maximum"
        )
    # Q is read with a plain global_load_vN (no buffer bound), and the epilogue stores
    # the same rows, so a query tile that runs past seqlen_q reads and writes out of
    # bounds. Implied by the dataclass (seqlen_q % 256 == 0) at the default block_m.
    if spec.seqlen_q % tuning.block_m != 0:
        return False, (
            f"seqlen_q={spec.seqlen_q} must be a multiple of "
            f"tuning.block_m={tuning.block_m}: the last query tile would otherwise "
            f"run past seqlen_q, and Q/O are addressed without a bounds check"
        )
    # Pads, in elements. The QK / cfvst-V reads are smem_load_vN(n=4) = 8-byte
    # payloads, which the lowering stamps `align 8` on unconditionally; the column
    # index inside a row is always a multiple of 4 elements, so keeping the PITCH a
    # multiple of 4 elements is what keeps that alignment contract true. A pitch that
    # breaks it keeps the ds_read and silently returns the wrong data or faults --
    # the same failure mode the spec's lds_k_group_pad % 8 check guards for ds_read_b128.
    for _pad_name, _pad in (
        ("lds_row_pad", tuning.lds_row_pad),
        ("v_row_pad", tuning.resolved_v_row_pad(spec)),
    ):
        if _pad < 0 or _pad % 4 != 0:
            return False, (
                f"tuning.{_pad_name} must be a non-negative multiple of 4 elements "
                f"(8 bytes) so the LDS pitch stays aligned for the n=4 ds_read, got "
                f"{_pad}"
            )
    _wpe = tuning.resolved_waves_per_eu(spec)
    if _wpe <= 0:
        return False, (
            f"resolved waves_per_eu must be positive, got {_wpe} (tuning."
            f"waves_per_eu={tuning.waves_per_eu}, spec.waves_per_eu="
            f"{spec.waves_per_eu})"
        )
    # cfvst forced ON where the policy says OFF. Both cases the policy excludes are
    # known-bad, not merely untuned: D64 (rows-per-DMA > 1) is VGPR-bound, so the
    # register round-trip regresses it AND the transposed store no longer matches the
    # packed async-DMA layout the naive path relies on; bf16 D128 spills past the
    # waves-per-eu cap on the .1k schedule. Turning cfvst OFF is always legal (it is
    # the naive path), so only the ON direction is gated. See _use_cfvst.
    if tuning.resolved_use_cfvst(spec) and not _use_cfvst(spec.head_size, spec.dtype):
        return False, (
            f"tuning.use_cfvst=True is rejected at D{spec.head_size}/{spec.dtype}: "
            f"the conflict-free-V store is measured-negative there (D64 is VGPR-bound "
            f"and regresses; bf16 D128 spills past the waves-per-eu cap), and it is "
            f"only tile-exact on the rows-per-DMA==1 fp16 path -- see _use_cfvst"
        )

    # --- Tile geometry. The causal KV-loop clamp uses n_per = block_m //
    # block_n, a FLOOR: a block_n that does not divide the query tile silently drops
    # every key past the last whole sub-tile, and block_n > block_m makes n_per 0
    # -> zero-trip loop -> l == 0 -> rcp(0) -> NaN. Neither fails loudly, so reject.
    if tuning.block_m % spec.block_n != 0:
        return False, (
            f"block_n must divide the {tuning.block_m}-row query tile (got "
            f"block_n={spec.block_n}; the spec also requires block_n % 32 == 0, so "
            f"use 32, 64, 128 or 256). Load-bearing for causal=True, where "
            f"n_per = {tuning.block_m} // block_n floors and drops keys; enforced "
            f"unconditionally so the two grids cannot diverge by a knob"
        )

    # --- Wave/tile divisibility, mirrored from the builder so support() and build()
    # agree on exactly one set of specs (the module contract at the top of this file).
    # The condition below is ALSO enforced in _build_attention_dense_single_buffer;
    # without it here, a spec support() accepted would die in the builder with a
    # ValueError --
    # precisely the dispatch fall-through hole this gate exists to close.
    _waves = tuning.block_m // 32
    _rpi = _rows_per_instr(spec.head_size)
    if spec.block_n % _waves != 0 or (spec.block_n // _waves) % _rpi != 0:
        return False, (
            f"block_n={spec.block_n} over {_waves} waves gives ROWS_PER_WAVE="
            f"{spec.block_n // _waves}, not a multiple of ROWS_PER_INSTR={_rpi} "
            f"(D={spec.head_size}); the async K/V DMA would skip rows"
        )
    # cfvst assigns one 2x2 (token-pair x dim-pair) block per thread with no tail
    # guard, so the CTA thread count must divide the block count. Gated on the cfvst
    # predicate: the naive-V path never reads these quantities, and checking it
    # unconditionally would reject specs the naive path builds fine.
    if tuning.resolved_use_cfvst(spec):
        _threads = _waves * 64
        _vblocks = (spec.block_n // 2) * (spec.head_size // 2)
        if _vblocks % _threads != 0:
            return False, (
                f"cfvst: V 2x2 block count {_vblocks} (block_n={spec.block_n}, "
                f"D={spec.head_size}) is not a multiple of the {_threads}-thread CTA"
            )

    # --- LDS budget. Without this, an over-budget tile reaches comgr and fails with
    # an opaque CODEGEN_BC_TO_RELOCATABLE abort instead of a structured reason.
    lds_bytes = _lds_bytes(spec, tuning)
    from ..common.attention_arch import attention_lds_capacity_bytes

    capacity = attention_lds_capacity_bytes(arch)
    if lds_bytes > capacity:
        return False, (
            f"K_lds+V_lds needs {lds_bytes} B at block_n={spec.block_n}, "
            f"D={spec.head_size}, which exceeds the {arch} LDS capacity ({capacity} B)"
        )

    # --- 32-bit addressing. Every offset below is built from IRBuilder add/mul, which
    # lower to `add nsw` / `mul nsw` i32 -- signed overflow is UB, not a wrap, so LLVM
    # may poison the whole address chain rather than merely read the wrong place. The
    # buffer-resource num_records field is unsigned in hardware, but it is emitted via
    # const_i32 (no range check) and the voffset feeding it is signed i32 arithmetic,
    # so the signed bound is the binding one on both paths.
    kv_bytes = spec.batch * spec.seqlen_kv * spec.num_kv_heads * spec.head_size * 2
    if kv_bytes >= _INT32_LIMIT:
        return False, (
            f"K/V extent is {kv_bytes} B, at or past the 32-bit buffer-resource "
            f"limit ({_INT32_LIMIT} B)"
        )
    qo_elems = spec.batch * spec.seqlen_q * spec.num_query_heads * spec.head_size
    if qo_elems >= _INT32_LIMIT:
        return False, (
            f"Q/O extent is {qo_elems} elements, at or past the 32-bit addressing "
            f"limit ({_INT32_LIMIT})"
        )
    return True, ""


def build_attention_dense(
    spec: AttentionDenseSpec,
    *,
    arch: str = "gfx942",
    tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING,
) -> KernelDef:
    """Emit the gfx942 dense flash-attention prefill kernel for ``spec``.

    Delegates to :func:`_build_attention_dense_single_buffer`, the single-LDS-buffer
    (NBUF=1) body (32x32x8 atom + K-loop doubling). NBUF=1 is the only axis that name
    describes: the body carries the full shipped lever set (cfvst, exp2_fast, fused
    rescale, per-config waves-per-eu, persistent grid-stride), so it is not a
    correctness-first stand-in for a later tuned body. Every scope restriction --
    including the modes deferred to later follow-ups -- lives in
    :func:`supports_attention_dense`, which is consulted here, so this function is a
    thin gate-plus-delegate and cannot reject a spec that ``supports`` accepted.

    ``tuning`` is the gfx942-private sweep struct (:class:`Gfx942DenseTuning`); its
    default is the shipped configuration, so an existing caller that never mentions it
    gets byte-identical IR and a byte-identical kernel name.

    :raises NotImplementedError: ``arch`` is not gfx942.
    :raises ValueError: ``spec``/``tuning`` is outside the supported set; the message
        carries :func:`supports_attention_dense`'s structured reason (out-of-scope
        dtype or head size, a mode deferred to a later phase, a ``block_n`` that does
        not divide the query tile, an illegal tuning override, an over-budget LDS
        footprint, or an extent past the 32-bit addressing limit).
    """
    if arch != "gfx942":
        raise NotImplementedError(
            f"kernels.gfx942.attention_dense is gfx942-only (got {arch})"
        )
    ok, why = supports_attention_dense(spec, arch=arch, tuning=tuning)
    if not ok:
        raise ValueError(f"unsupported gfx942 attention_dense spec: {why}")
    return _build_attention_dense_single_buffer(spec, tuning)


def _build_attention_dense_single_buffer(
    spec: AttentionDenseSpec, tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING
) -> KernelDef:
    """gfx942 dense prefill body: 32x32x8 atom + K-loop doubling.

    Non-pipelined (NBUF=1): per KV tile load→wait→QK→mask→softmax→PV, single LDS
    buffer, simple online-softmax rescale. The per-work-item compute is factored into
    ``_run_work_item(qb, hq, bt)``, shared by the default grid (one CTA per work item)
    and the P4 persistent grid-stride path (one CTA strides over many). Perf levers
    are all on: conflict-free V/cfvst (P1, D128 fp16), exp2_fast + fused rescale (P2),
    per-config waves-per-eu (P3), persistent grid-stride + qb/hkv-major decode (P4).
    The transposed-QK architecture (S^T=K@Q^T so P feeds PV lane-locally) is kept
    from the gfx950 sibling — only the MFMA atom and K/V fragment widths change.
    """
    B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    Hq, Hkv, D = spec.num_query_heads, spec.num_kv_heads, spec.head_size
    causal = spec.causal
    dtype = _DTYPE_IR[spec.dtype]

    BLOCK_M = tuning.block_m  # _BLOCK_M at the shipped default
    WAVES = BLOCK_M // 32  # 8
    BN = spec.block_n

    N_SUB = BN // 32  # key 32-tiles per KV tile
    D_TILES = D // 32  # head-dim 32-tiles
    K_STEPS = D // 8  # QK K=8 doubled steps (was D//16 on gfx950)
    KK_STEPS = BN // 8  # PV K=8 doubled steps (was BN//16)
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D
    ROWS_PER_INSTR = _rows_per_instr(D)  # 1 for D128, 2 for D64

    b = IRBuilder(gfx942_kernel_name(spec, tuning))
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    # The RESOLVED waves-per-eu, so the emitted attribute and the wpe{N} token in the
    # name above are the same number by construction (a name/binary mismatch here is
    # what makes the name-keyed launcher cache serve a stale HSACO).
    b.kernel.attrs["waves_per_eu"] = tuning.resolved_waves_per_eu(spec)

    q = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k = b.param(
        "k_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    v = b.param(
        "v_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    o = b.param(
        "o_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    scale = b.param("scale", F32)
    qk_scale = b.fmul(scale, b.const_f32(LOG2E))

    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    lane_h = b.div(lane, b.const_i32(32))
    d_base = b.mul(lane_h, b.const_i32(4))  # K=8 half-split: lane_h in {0,1} -> col 0/4
    neg_inf = b.const_f32(-1e30)

    # ---- CTA-invariant LDS + buffer setup (allocated ONCE per CTA; for the
    # persistent grid it is REUSED across every work item the CTA strides over, and
    # for the default grid it is the single tile) ----
    # Padded for D128 (one row per DMA instr), unpadded for D64 (two rows per
    # instr need a contiguous stride) -- see _lds_row_stride. Shared with the
    # LDS-budget check in supports_attention_dense so the two cannot drift. The one
    # exception is when _k_group_pad_active(spec): D64 K_lds then takes the padded
    # [1, block_n // rows_per_instr, _k_group_stride] 2-row-group layout below.
    USE_CFVST = tuning.resolved_use_cfvst(spec)  # P1 conflict-free V: D128 fp16 only
    LDROW = _lds_row_stride(D, tuning)
    # Hypothesis #3 (D64 K-LDS bank conflicts): K_lds with a row-group boundary pad,
    # sized by the shared spec.lds_k_group_pad (0 disables it -- that is the A/B probe).
    KPAD_D64 = _k_group_pad_active(spec, tuning)
    if KPAD_D64:
        K_GROUP = ROWS_PER_INSTR  # rows per async-DMA instruction (=2 at D64)
        K_GROUP_STRIDE = _k_group_stride(spec, tuning)  # rows_per_instr*D + pad elems
        K_lds = b.smem_alloc(
            dtype, [1, BN // K_GROUP, K_GROUP_STRIDE], name_hint="Klds"
        )
    else:
        K_lds = b.smem_alloc(dtype, [1, BN, LDROW], name_hint="Klds")
    if USE_CFVST:
        # V stored TRANSPOSED (P1 conflict-free V): [dim, token] with token inner and
        # padded by resolved_v_row_pad, so the PV A-operand read is a contiguous
        # ds_read_b64 rather than P0's 4 element-wise ds_read_u16. Filled by the
        # perm_b32 store path below. This is one of exactly TWO reads that SIZE V_lds --
        # the other is the _lds_bytes budget -- and both take it from the same
        # resolved struct, so the budget cannot under-count this allocation.
        V_LDROW = BN + tuning.resolved_v_row_pad(spec)
        V_lds = b.smem_alloc(dtype, [1, D, V_LDROW], name_hint="VldsT")
        # V^T bank-conflict swizzle (col' = key ^ ((dim&31)<<2)): needs a pow2 row
        # width >= 128 so the XOR stays in-bounds. Gated OFF otherwise (default
        # v_row_pad=8 -> width 72 -> no swizzle, original behavior preserved).
        SWZ_V = (V_LDROW & (V_LDROW - 1)) == 0 and V_LDROW >= 128
    else:
        # V keeps the natural [token, dim] async-DMA layout (D64: VGPR-bound, cfvst
        # regresses it -- see _use_cfvst); read element-wise in read_v.
        V_lds = b.smem_alloc(dtype, [1, BN, LDROW], name_hint="Vlds")

    n_ktiles = Skv // BN
    n_per = BLOCK_M // BN

    # ---- async DMA loaders (arch-neutral; width=1 = CDNA3 legal) ----
    K_LDROW_BYTES = LDROW * 2
    ROWS_PER_WAVE = BN // WAVES
    # The D64 loader steps ROWS_PER_INSTR rows per DMA instruction with a FLOOR
    # (`range(ROWS_PER_WAVE // ROWS_PER_INSTR)`), and one instruction writes
    # 64 lanes x 4 B = 2 unpadded D64 rows starting at row0*K_LDROW_BYTES -- so row0
    # must be even. A non-multiple would silently skip the tail rows, leaving
    # uninitialized LDS that do_qk/read_v then consume as garbage (no fault). Holds
    # today because block_n % 32 == 0 makes BN//8 a multiple of 4; bind it rather
    # than rely on the dataclass validator staying that strict.
    # Raised, not asserted: `python -O` strips asserts, and skipped DMA rows are a
    # SILENT wrong answer (uninitialized LDS consumed as garbage, no fault) -- the
    # same reasoning as the `_BLOCK_M % 32` wave-coverage guard at module scope.
    # Mirrored in supports_attention_dense so support() and build() cannot disagree.
    if BN % WAVES != 0 or ROWS_PER_WAVE % ROWS_PER_INSTR != 0:
        raise ValueError(
            f"block_n={BN} over {WAVES} waves gives ROWS_PER_WAVE={ROWS_PER_WAVE}, "
            f"not a multiple of ROWS_PER_INSTR={ROWS_PER_INSTR} (D={D}); the async "
            f"DMA would skip rows"
        )
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    # cfvst feeds V via buffer_load + perm_b32 + smem_store (no async-DMA handle); the
    # naive D64 path still lands V through async_buffer_load_lds, which needs the base.
    V_lds_addr = None if USE_CFVST else b.smem_addr_of(V_lds)
    k_rsrc = b.buffer_rsrc(k, b.const_i32(B * Skv * Hkv * D * 2))
    v_rsrc = b.buffer_rsrc(v, b.const_i32(B * Skv * Hkv * D * 2))

    # ---- conflict-free V (P1): perm_b32 store-path transpose into V_lds[dim, token] ----
    # Load V naturally [token, dim] (coalesced VMEM over the contiguous dim axis),
    # transpose each 2x2 f16 block in-thread with perm_b32 (no cross-lane, no lgkmcnt;
    # CK transpose_vectors masks), then write ONE contiguous 2-half ds_write per dim
    # row into V_lds[dim, token]. The consumer read_v then reads 4 consecutive tokens
    # at a fixed dim as a single ds_read_b64. Lifted from the proven
    # attention_tiled_2d cfvst vehicle, after a cross-part correctness disproof of its
    # parked sign-flip symptom (procedure recorded in the optimization plan).
    # The block loop is statically unrolled (V_ITEMS is small: 4 at D128/BN64), which
    # is well under the tiled_2d full-unroll IR-explosion threshold.
    THREADS = WAVES * 64
    if USE_CFVST:
        V_TOK_PAIRS = BN // 2
        V_DIM_PAIRS = D // 2
        V_BLOCKS = V_TOK_PAIRS * V_DIM_PAIRS
        # tid-strided block assignment must tile the 2x2 grid exactly (no partial
        # last item, so no per-item guard).
        #
        # CFVST-PATH-ONLY. Computing this unconditionally would reject specs the
        # NAIVE V path builds fine: at D64/block_n=32 with a 1024-thread CTA,
        # V_BLOCKS=512 < THREADS=1024 and the check fires even though `read_v` never
        # touches V_BLOCKS there (D64 is naive-V by _use_cfvst). Raised, not
        # asserted, for the reason stated at the DMA row-split guard above:
        # `python -O` strips asserts, and this one guards a silent wrong answer (an
        # unguarded tail item would read past the tile).
        if V_BLOCKS % THREADS != 0:
            raise ValueError(
                f"cfvst: V 2x2 block count {V_BLOCKS} (block_n={BN}, D={D}) is not a "
                f"multiple of THREADS={THREADS}; a tail item would need a bounds guard"
            )
        V_ITEMS = V_BLOCKS // THREADS
    else:
        V_TOK_PAIRS = V_DIM_PAIRS = V_BLOCKS = V_ITEMS = 0

    def _cfvst_store_v(payload):
        """perm_b32 2x2 transpose + contiguous ds_write publishing V_lds[dim, token].

        CTA-invariant: reads only the per-tile payload and writes V_lds (a fixed
        allocation), so it is shared by every work item."""
        for d0, t0, x0, x1 in payload:
            # Each output i32 holds 2 CONSECUTIVE tokens (t0, t0+1) at one fixed dim.
            row_d0 = b.perm_b32(x0, x1, b.const_i32(0x01000504))  # (V[t0,d0], V[t1,d0])
            row_d1 = b.perm_b32(
                x0, x1, b.const_i32(0x03020706)
            )  # (V[t0,d0+1], V[t1,d0+1])
            d1 = b.add(d0, b.const_i32(1))
            if SWZ_V:
                c0 = b.xor(t0, b.shl(b.land(d0, b.const_i32(31)), b.const_i32(2)))
                c1 = b.xor(t0, b.shl(b.land(d1, b.const_i32(31)), b.const_i32(2)))
            else:
                c0, c1 = t0, t0
            b.smem_store_vN(
                V_lds,
                [b.const_i32(0), d0, c0],
                b.bitcast(row_d0, VectorType(dtype, 2)),
                2,
            )
            b.smem_store_vN(
                V_lds,
                [b.const_i32(0), d1, c1],
                b.bitcast(row_d1, VectorType(dtype, 2)),
                2,
            )

    def read_v(dt, kk):
        """PV A-operand = V^T[dim, key]. dim=dt*32+lane_m, keys=kk*8+lane_h*4+{0..3}.

        cfvst (D128): the 4 keys are CONTIGUOUS in the transposed V_lds[dim, token],
        so this is ONE ds_read_b64 vs the naive path's 4 element-wise ds_read_u16.
        The values delivered to the MFMA are bit-identical between the two paths (same
        (dim, key) mapping); only the LDS layout and read width differ.
        naive (D64 / bf16-D128): element-wise from V_lds[key, dim] (bank-heavy, but
        those configs are VGPR-bound / spill under cfvst, not LDS-read-bound)."""
        if USE_CFVST:
            dim_row = b.add(b.const_i32(dt * 32), lane_m)
            key0 = b.add(b.const_i32(kk * 8), d_base)
            col = key0
            if SWZ_V:
                col = b.xor(
                    key0, b.shl(b.land(dim_row, b.const_i32(31)), b.const_i32(2))
                )
            return b.smem_load_vN(V_lds, b.const_i32(0), dim_row, col, dtype=dtype, n=4)
        dim_col = b.add(b.const_i32(dt * 32), lane_m)
        elems = []
        for j in range(4):
            key = b.add(b.add(b.const_i32(kk * 8), d_base), b.const_i32(j))
            vv = b.smem_load_vN(V_lds, b.const_i32(0), key, dim_col, dtype=dtype, n=1)
            elems.append(b.vec_extract(vv, 0))
        return b.vec_pack(elems, dtype)

    def do_pv(o_acc_in, p_packs):
        out = []
        for dt in range(D_TILES):
            acc_o = o_acc_in[dt]
            for kk in range(KK_STEPS):
                acc_o = mfma_32x32x8_for_dtype(
                    b, dtype, read_v(dt, kk), p_packs[kk], acc_o
                )
            out.append(acc_o)
        return out

    def _run_work_item(qb, hq, bt):
        """Emit the full attention for ONE (query-block, query-head, batch) work item.

        The default grid calls this once with the CTA's block ids; the persistent grid
        calls it once per grid-stride iteration with the decoded work item. Everything
        that depends on the work-item coordinates lives here (Q/K base offsets, the Q
        packs, the tile loaders, the causal clamp, the KV loop, the O epilogue); the
        LDS buffers, buffer resources, and the coordinate-free helpers above are
        CTA-invariant and closed over."""
        hkv = b.div(hq, b.const_i32(gqa))
        q_tok0 = b.add(b.mul(qb, b.const_i32(BLOCK_M)), b.mul(wave, b.const_i32(32)))
        q_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq, b.const_i32(D)),
        )
        k_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Skv)), b.const_i32(stride_k_tok)),
            b.mul(hkv, b.const_i32(D)),
        )

        # Q packs (QK B-operand), scaled once by qk_scale so exp2(s) is direct.
        q_tok = b.add(q_tok0, lane_m)
        q_packs = []
        for ks in range(K_STEPS):
            col = b.add(b.const_i32(ks * 8), d_base)
            addr = b.add(b.add(q_base, b.mul(q_tok, b.const_i32(stride_q_tok))), col)
            raw = b.global_load_vN(q, addr, dtype, 4, align=8)
            elems = [
                b.cast_f32_to(
                    b.fmul(b.cast_to_f32(b.vec_extract(raw, j)), qk_scale), dtype
                )
                for j in range(4)
            ]
            q_packs.append(b.vec_pack(elems, dtype))

        def _async_load(rsrc, lds_base, tile_key0, group_pad=False):
            if ROWS_PER_INSTR == 1:
                for r in range(ROWS_PER_WAVE):
                    row = b.add(b.mul(wave, b.const_i32(ROWS_PER_WAVE)), b.const_i32(r))
                    row_lds_off = b.zext(b.mul(row, b.const_i32(K_LDROW_BYTES)), I64)
                    row_base = b.smem_ptr_add(lds_base, row_lds_off)
                    gkey = b.add(tile_key0, row)
                    gcol = b.mul(lane, b.const_i32(2))
                    voff = b.add(
                        b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), gcol
                    )
                    b.async_buffer_load_lds_addr(
                        rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                    )
            else:
                lanes_per_row = D // 2
                sub_row = b.div(lane, b.const_i32(lanes_per_row))
                col = b.mul(b.mod(lane, b.const_i32(lanes_per_row)), b.const_i32(2))
                for it in range(ROWS_PER_WAVE // ROWS_PER_INSTR):
                    row0 = b.add(
                        b.mul(wave, b.const_i32(ROWS_PER_WAVE)),
                        b.const_i32(it * ROWS_PER_INSTR),
                    )
                    if group_pad:
                        # 2-row-group boundary pad (the D64 K-pad probe): the DMA still
                        # writes one whole group (ROWS_PER_INSTR contiguous rows)
                        # per instruction, but consecutive groups are spaced by the
                        # padded K_GROUP_STRIDE so do_qk's krow reads land 4-way (not
                        # 32-way) bank-conflicted. group = row0 // ROWS_PER_INSTR.
                        grp = b.add(
                            b.mul(
                                wave,
                                b.const_i32(ROWS_PER_WAVE // ROWS_PER_INSTR),
                            ),
                            b.const_i32(it),
                        )
                        row_lds_off = b.zext(
                            b.mul(grp, b.const_i32(K_GROUP_STRIDE * 2)), I64
                        )
                    else:
                        row_lds_off = b.zext(
                            b.mul(row0, b.const_i32(K_LDROW_BYTES)), I64
                        )
                    row_base = b.smem_ptr_add(lds_base, row_lds_off)
                    gkey = b.add(b.add(tile_key0, row0), sub_row)
                    voff = b.add(
                        b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), col
                    )
                    b.async_buffer_load_lds_addr(
                        rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                    )

        def load_tile(tile_idx):
            # cfvst (D128): K only; V is fed through the perm_b32 store path below
            # (async DMA would land V in the WRONG [token, dim] orientation for a
            # conflict-free read). Naive (D64): K and V both via async DMA, natural
            # [token, dim].
            tk0 = b.mul(tile_idx, b.const_i32(BN))
            _async_load(k_rsrc, K_lds_addr, tk0, group_pad=KPAD_D64)
            if not USE_CFVST:
                _async_load(v_rsrc, V_lds_addr, tk0)

        def _cfvst_load_v(tile_idx):
            """Issue the cfvst VMEM loads and keep each thread's V tile in VGPRs.

            Returns a payload of ``(d0, t0, x0, x1)`` per unrolled item: ``x0``/``x1``
            are the i32-packed ``<2 x elem>`` loads of token rows ``t0``/``t0+1`` at
            the contiguous dim pair ``(d0, d0+1)``. Loads go through ``v_rsrc`` so an
            out-of-range token is hardware-clamped to 0 (matches the async path)."""
            tile_tok0 = b.mul(tile_idx, b.const_i32(BN))
            payload = []
            for it in range(V_ITEMS):
                blk = b.add(b.mul(b.const_i32(it), b.const_i32(THREADS)), tid)
                # dim-pair is the fastest-varying coord so adjacent lanes issue
                # coalesced VMEM loads over the natural [token, dim] layout.
                tg = b.div(blk, b.const_i32(V_DIM_PAIRS))
                dg = b.mod(blk, b.const_i32(V_DIM_PAIRS))
                t0 = b.mul(tg, b.const_i32(2))
                d0 = b.mul(dg, b.const_i32(2))
                gk0 = b.add(tile_tok0, t0)
                gk1 = b.add(gk0, b.const_i32(1))
                # byte offset of (token, d0); contiguous dim pair -> one 2-half load.
                eoff0 = b.add(b.add(k_base, b.mul(gk0, b.const_i32(stride_k_tok))), d0)
                eoff1 = b.add(b.add(k_base, b.mul(gk1, b.const_i32(stride_k_tok))), d0)
                x0 = b.buffer_load_vN(
                    v_rsrc, b.mul(eoff0, b.const_i32(2)), zero_soff, dtype, 2
                )
                x1 = b.buffer_load_vN(
                    v_rsrc, b.mul(eoff1, b.const_i32(2)), zero_soff, dtype, 2
                )
                payload.append((d0, t0, b.bitcast(x0, I32), b.bitcast(x1, I32)))
            return payload

        def do_qk():
            """S^T = K@Q^T via the 32x32x8 atom (K-doubled)."""
            s_reg = []
            for nsub in range(N_SUB):
                acc = b.zero_vec_f32(16)
                krow = b.add(b.const_i32(nsub * 32), lane_m)
                if KPAD_D64:
                    # 2-row-group padded K_lds[1, BN//2, 2*D+pad]: decompose the
                    # logical krow into (group = krow >> 1, within = krow & 1); the
                    # padded group stride spreads consecutive krow across 4 (not 32)
                    # banks (the D64 K-pad probe). col stays within the D-wide row.
                    k_group = b.lshr(krow, b.const_i32(1))
                    k_within_off = b.mul(b.land(krow, b.const_i32(1)), b.const_i32(D))
                for ks in range(K_STEPS):
                    col = b.add(b.const_i32(ks * 8), d_base)
                    if KPAD_D64:
                        k_pack = b.smem_load_vN(
                            K_lds,
                            b.const_i32(0),
                            k_group,
                            b.add(k_within_off, col),
                            dtype=dtype,
                            n=4,
                        )
                    else:
                        k_pack = b.smem_load_vN(
                            K_lds, b.const_i32(0), krow, col, dtype=dtype, n=4
                        )
                    acc = mfma_32x32x8_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                s_reg.append([b.vec_extract(acc, i) for i in range(16)])
            return s_reg

        def do_mask(s_reg, tile_idx):
            if not causal:
                return
            tile_key0 = b.mul(tile_idx, b.const_i32(BN))
            query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
            for nsub in range(N_SUB):
                sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
                for i in range(16):
                    ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                    s_reg[nsub][i] = b.select(
                        b.cmp_le(ktok, query_tok), s_reg[nsub][i], neg_inf
                    )

        # n_up: causal clamps the KV loop to the diagonal tile of this query block.
        n_ktiles_c = b.const_i32(n_ktiles)
        if causal:
            n_up = b.add(b.mul(qb, b.const_i32(n_per)), b.const_i32(n_per))
            n_up = b.select(b.cmp_lt(n_up, n_ktiles_c), n_up, n_ktiles_c)
        else:
            n_up = n_ktiles_c

        # ---- online-softmax main loop (non-pipelined, single buffer) ----
        m0 = neg_inf
        l0 = b.const_f32(0.0)
        o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
        iter_args = [("m", m0), ("l", l0)] + [
            (f"o{dt}", o0[dt]) for dt in range(D_TILES)
        ]

        # elide_trailing_barrier=False: the trailing barrier is NOT an optimizable
        # rendezvous -- it is the WAR guard on the SINGLE K/V LDS buffer. The elide
        # pass (lower_llvm._lower_unrolled_for) targets body_ops[-2], which is exactly
        # this barrier's slot, and only misses it today because the op-name match is
        # hardcoded to "tile.sync" and unroll defaults False. Pin it so a future
        # unroll=True (P3) or a sync_lds_only->sync swap cannot silently delete it.
        # Verified byte-identical codegen with and without the flag today.
        loop = b.scf_for_iter(
            b.const_i32(0),
            n_up,
            b.const_i32(1),
            iter_args,
            iv_name="kt",
            elide_trailing_barrier=False,
        )
        with loop as (j, carry):
            m_i = carry[0]
            l_i = carry[1]
            o_acc = list(carry[2 : 2 + D_TILES])

            if tuning.iglp:
                # Runbook lever 7: one canned-scheduler hint at the loop-body top.
                # Only meaningful on the cfvst path (in-loop ds_write to interleave).
                b.iglp_opt(0)

            if USE_CFVST:
                load_tile(j)  # K async DMA -> K_lds
                v_payload = _cfvst_load_v(
                    j
                )  # V register buffer_loads (natural [token,dim])
                # Drain BOTH the K async DMA and the V register loads (all vmcnt)
                # before the V perm+store reads those registers and before QK reads
                # K_lds.
                b.s_waitcnt(vmcnt=0)
                _cfvst_store_v(v_payload)  # perm_b32 -> ds_write V_lds[dim,token]
                # Publish K_lds (DMA) AND V_lds (ds_write) to all waves. This MUST
                # drain lgkmcnt for the V ds_write and cannot be a bare s_barrier:
                # gfx90a+ set FeatureBackOffBarrier, so LLVM would sink the lgkm wait
                # past the barrier to the ds_read consumer (the P0 read-after-barrier
                # race, now on the tile-START barrier because V publication is an
                # in-loop ds_write). sync_lds_only() emits the lgkm drain BEFORE the
                # barrier (CK Tile block_sync_lds). vmcnt is already 0, no full sync.
                b.sync_lds_only()
            else:
                # Naive (D64): K and V both async-DMA'd into their natural layout.
                # Drain the DMA (vmcnt) then a plain barrier -- V is not an in-loop
                # ds_write here, so the tile-start barrier needs no lgkm drain (the
                # DMA landed the data; the trailing sync_lds_only guards the
                # read-before-overwrite).
                load_tile(j)
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare()

            s = do_qk()
            do_mask(s, j)

            # tile max over keys (both lane-halves) for this query.
            local_max = neg_inf
            for nsub in range(N_SUB):
                for i in range(16):
                    local_max = b.fmax(local_max, s[nsub][i])
            tile_max = b.fmax(local_max, b.warp_shuffle_xor(local_max, 32))
            m_new = b.fmax(m_i, tile_max)
            # P2: exp2_fast (llvm.amdgcn.exp2.f32 -> one v_exp_f32) drops the ~5-VALU
            # guarded range reduction that plain exp2 (llvm.exp2.f32) emits. Safe
            # here: both softmax args are always <= 0 -- alpha's m_i - m_new (m_new =
            # max(m_i, tile_max) >= m_i) and p's s - m_new (m_new >= tile_max >= every
            # s) -- exactly exp2_fast's precondition (no overflow; v_exp_f32 flushes
            # large negatives to 0). Cuts ~99 VALU/tile at D128, the dominant
            # MFMA-starving residual once conflict-free V (P1) lands. Enabled for
            # every config except bf16 D128, which spills on the .1k schedule -- the
            # spill rationale and the matrix live in _use_exp2_fast's docstring.
            exp2 = b.exp2_fast if tuning.resolved_use_exp2_fast(spec) else b.exp2
            alpha = exp2(b.fsub(m_i, m_new))

            # P2 fused/lazy rescale: compute each exp2 inline, accumulate l_local, and
            # cast->pack it into the PV B-operand in ONE pass. The f32 p value dies
            # after its two uses (the l_local fadd + the dtype cast) instead of
            # staying live across the whole l_local reduction AND a separate cast/pack
            # pass. Peak live f32 p regs drop from N_SUB*16 (=32 at D128) to ~one
            # pack, freeing ~28 VGPR (plan §6.1).
            #
            # The kk->(nsub,i) map is the PV B-operand relayout (key=kk*8+lane_h*4+j
            # maps to QK C-reg nsub=kk//4, i=(kk%4)*4+j -- same lane_h split, no
            # cross-lane). KK_STEPS==N_SUB*4, so this covers every p element exactly
            # once, and i steps 0,1,2,3 / 4,5,6,7 / ... within each nsub -- the
            # identical accumulation order as a nested (nsub,i) loop, so l_local and
            # the packed casts are numerically bit-identical: pure live-range relief.
            l_local = b.const_f32(0.0)
            p_packs = []
            for kk in range(KK_STEPS):
                nsub = kk // 4
                base_i = (kk % 4) * 4
                elems = []
                for j2 in range(4):
                    pv = exp2(b.fsub(s[nsub][base_i + j2], m_new))
                    l_local = b.fadd(l_local, pv)
                    elems.append(b.cast_f32_to(pv, dtype))
                p_packs.append(b.vec_pack(elems, dtype))
            l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            l_new = b.fadd(b.fmul(l_i, alpha), l_tile)

            o_acc = [
                b.vec_pack(
                    [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)], F32
                )
                for dt in range(D_TILES)
            ]
            o_acc = do_pv(o_acc, p_packs)
            # Tile done; the next iteration refills the SINGLE K/V buffer (K via async
            # DMA, V via the perm_b32 ds_write), so every wave's LDS reads must have
            # LANDED (not just issued) before any wave starts writing. This MUST drain
            # lgkmcnt and cannot be a bare s_barrier: gfx90a+ set FeatureBackOffBarrier,
            # so SIInsertWaitcnts skips the conservative pre-barrier drain and places
            # the lgkm wait at the ds_read's CONSUMER -- which the scheduler is free to
            # sink past the barrier. Measured before the fix: ds_read_u16 x4 ->
            # s_barrier -> s_waitcnt lgkmcnt(0). sync_lds_only() emits the drain BEFORE
            # the barrier (CK Tile block_sync_lds); real LDS ops (mayLoad/mayStore)
            # cannot be scheduled across the side-effecting s_waitcnt, so the window is
            # structurally closed. vmcnt is deliberately NOT drained: it is already 0
            # (the tile-start s_waitcnt(vmcnt=0) drained this tile's DMA and NBUF=1
            # issues no new DMA until the next iteration), so a full sync() would add a
            # dead instruction. NOTE: the gfx950 sibling is NOT a precedent for a bare
            # barrier here. Its NBUF=2 separates K (read j%2, written (j+1)%2) but NOT
            # V: it reads vbuf_prev=(j+1)%2 and writes pbuf=(j+1)%2 -- the SAME buffer,
            # across a bare s_barrier_bare(). Correct there only by scheduler luck;
            # tracked separately, do not copy that idiom.
            b.sync_lds_only()
            b.scf_yield(m_new, l_new, *o_acc)

        res = loop.results
        l_i = res[1]
        o_acc = list(res[2 : 2 + D_TILES])

        # Epilogue: O[query,dim] = (P@V)/l, from the transposed C[dim,query] accum.
        rcp_l = b.rcp(l_i)
        o_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq, b.const_i32(D)),
        )
        qtok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        q_row_byte = b.add(o_base, b.mul(qtok, b.const_i32(stride_q_tok)))
        d_half = b.mul(lane_h, b.const_i32(4))
        for dt in range(D_TILES):
            for g in range(4):
                d0 = b.add(b.const_i32(dt * 32 + g * 8), d_half)
                addr = b.add(q_row_byte, d0)
                vals = [
                    b.cast_f32_to(
                        b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l), dtype
                    )
                    for kk in range(4)
                ]
                b.global_store_vN(o, addr, b.vec_pack(vals, dtype), 4, align=8)

    # ---- grid dispatch: default (one CTA per work item) vs persistent (P4) ----
    if spec.persistent:
        # 1-D grid of NP long-lived CTAs; each grid-strides over the flattened work
        # space W = NQB*Hq*B, amortizing per-CTA launch + scalar setup + the cold LDS
        # prime across many work items instead of once per query block. The inner
        # per-work-item compute is byte-identical to the default path (same
        # _run_work_item); only the outer loop + work decode + cross-item LDS
        # rendezvous are new. Ported from the gfx950 sibling.
        NP = spec.num_persistent
        NQB = Sq // BLOCK_M  # ragged rejected -> Sq % BLOCK_M == 0, exact
        W = NQB * Hq * B
        cta_id = b.block_id_x()
        outer = b.scf_for(cta_id, b.const_i32(W), b.const_i32(NP), iv_name="wi")
        with outer as wi:
            # Cross-work-item LDS reuse guard. The previous item's epilogue touches
            # only registers + the O buffer, and its last KV tile's trailing
            # sync_lds_only was the final LDS-read barrier -- so lgkm is already
            # drained. Re-sync all waves (and drain the epilogue O store via vmcnt=0)
            # before this item reissues loads into the shared K/V buffers; a bare
            # barrier suffices because there is no pending lgkm to sink here.
            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare()
            if spec.resolved_persist_decode == "hkv_major":
                # hkv-MAJOR + causal-balanced decode (gfx950 §):
                #   wi = hkv*(NQB*gqa*B) + blk*(gqa*B) + hql*B + bt
                # hkv in the MSB keeps each grid-stride phase within ~1 kv-head so the
                # shared GQA K/V stays L2-resident across its gqa query heads; blk is
                # folded so a CTA striding both halves of a kv-head does qb=X and
                # qb=NQB-1-X (constant causal cost) -- qb_major's balance + L2 win.
                half = NQB // 2
                bt_v = b.mod(wi, b.const_i32(B))
                rem = b.div(wi, b.const_i32(B))
                hql = b.mod(rem, b.const_i32(gqa))
                r2 = b.div(rem, b.const_i32(gqa))
                blk = b.mod(r2, b.const_i32(NQB))
                hkv_wi = b.div(r2, b.const_i32(NQB))
                hq_v = b.add(b.mul(hkv_wi, b.const_i32(gqa)), hql)
                qb_hi = b.sub(b.const_i32(NQB - 1 + half), blk)  # NQB-1-(blk-half)
                qb_v = b.select(b.cmp_lt(blk, b.const_i32(half)), blk, qb_hi)
            else:
                # qb-MAJOR decode: wi = qb*(Hq*B) + hq*B + bt. Putting qb (the
                # triangular causal-cost index) in the MSB spreads cheap+expensive
                # query blocks across each CTA under grid-stride. Optional interleave
                # flips on ODD `rem` (= qb0*Hq + hq, so it alternates per-hq within a
                # qb0 row, NOT per qb0) to further balance the causal tail.
                bt_v = b.mod(wi, b.const_i32(B))
                rem = b.div(wi, b.const_i32(B))
                hq_v = b.mod(rem, b.const_i32(Hq))
                qb0 = b.div(rem, b.const_i32(Hq))
                if spec.interleave and causal and NQB > 1:
                    odd = b.cmp_eq(b.mod(rem, b.const_i32(2)), b.const_i32(1))
                    qb_v = b.select(odd, b.sub(b.const_i32(NQB - 1), qb0), qb0)
                else:
                    qb_v = qb0
            _run_work_item(qb_v, hq_v, bt_v)
    else:
        _run_work_item(b.block_id_x(), b.block_id_y(), b.block_id_z())
    b.ret()
    return b.kernel


# --- public geometry / ABI surface (arch-neutral; mirrors the gfx950 helpers) ---


def attention_dense_grid(
    spec: AttentionDenseSpec, tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING
) -> tuple[int, int, int]:
    """Launch grid: persistent = 1-D grid of ``num_persistent`` CTAs; default =
    one CTA per (query-block, query-head, batch).

    Sized from ``tuning.block_m`` (``_BLOCK_M`` at the default) so the grid and the
    body's query tiling cannot disagree -- a mismatch writes some rows twice and
    others never."""
    if spec.persistent:
        return (spec.num_persistent, 1, 1)
    # ceil kept for parity with the gfx950 helper; on gfx942 it is always exact,
    # because ragged is rejected and supports_attention_dense then enforces
    # seqlen_q % block_m == 0.
    nqb = (spec.seqlen_q + tuning.block_m - 1) // tuning.block_m
    return (nqb, spec.num_query_heads, spec.batch)


def attention_dense_block(
    spec: AttentionDenseSpec, tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING
) -> tuple[int, int, int]:
    """CTA block dims: ``tuning.block_m // 32`` wave64s.

    Derived from the tuning field rather than ``spec.num_waves`` (which the gfx950
    spec hardcodes to ``_BLOCK_M // 32``) so a block_m sweep point launches the
    thread count the body actually emits; identical at the default."""
    del spec  # geometry is fully determined by block_m
    return (tuning.block_m // 32 * 64, 1, 1)


def attention_dense_signature(spec: AttentionDenseSpec):
    """ABI signature: q/k/v/o pointers + f32 scale.

    THE single definition of this kernel's ABI -- the builder and the benchmark both
    call it rather than re-deriving the parameter list, so a reordering cannot drift
    between the emitted kernel and the launcher packing arguments for it.

    No ``cu_seqlens`` pair: varlen is rejected by :func:`supports_attention_dense`, so
    there is no varlen kernel to describe. Those pointers land when varlen does.
    """
    from rocke.helpers.spec import SignatureBuilder

    return (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("v_ptr", spec.dtype)
        .ptr("o_ptr", spec.dtype)
        .scalar("scale", "f32")
        .build()
    )


_DENSE_LAUNCHER_CACHE: dict = {}


def run_attention_dense_torch(
    *,
    spec: AttentionDenseSpec,
    q,
    k,
    v,
    out,
    scale: float,
    stream: int = 0,
    arch: str = "gfx942",
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    tuning: "Gfx942DenseTuning" = _DEFAULT_TUNING,
):
    """High-level framework entry: compile (cached) + launch the gfx942 dense prefill
    kernel on torch tensors. ``q``/``out`` are ``[B, S, Hq, D]`` and ``k``/``v`` are
    ``[B, Skv, Hkv, D]``, dense contiguous; ``scale`` is the softmax scale (1/sqrt(D)).
    Returns ``out``. torch is imported lazily by the launcher, so this module stays
    torch-free at import time. Serves both the default and the P4 persistent grid
    (``spec.persistent``) -- ``attention_dense_grid`` picks the right launch shape.

    Mirrors ``kernels.gfx950.attention_dense.run_attention_dense_torch`` but keys the
    launcher cache on :func:`gfx942_kernel_name` (not the shared ``kernel_name()``):
    this kernel bakes ``batch`` into the buffer-resource extents and ``waves_per_eu``
    into the register-allocation attribute, both of which ``kernel_name()`` omits -- so
    two specs differing only in batch (or wpe) MUST NOT share a cached binary, or a B>1
    launch is served the B=1 kernel and reads out of bounds.

    varlen / ragged are rejected by :func:`supports_attention_dense` on gfx942, so the
    ABI is always the 5-arg (q, k, v, o, scale) form; passing ``cu_seqlens_*`` is a
    caller error rather than a silently-ignored argument.

    ``tuning`` (:class:`Gfx942DenseTuning`) defaults to the shipped configuration and
    is folded into the cache key through :func:`gfx942_kernel_name`, so two tunings
    that emit different IR can never share a cached launcher."""
    ok, why = supports_attention_dense(spec, arch=arch, tuning=tuning)
    if not ok:
        raise NotImplementedError(f"attention_dense unsupported for spec: {why}")
    if cu_seqlens_q is not None or cu_seqlens_kv is not None:
        raise ValueError(
            "cu_seqlens_* provided but gfx942 attention_dense is dense-only (varlen "
            "is rejected by supports_attention_dense); the ABI has no cu_seqlens args"
        )
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    # batch-, wpe- and tuning-unique cache key (see docstring): gfx942_kernel_name,
    # not the shared kernel_name.
    key = gfx942_kernel_name(spec, tuning)
    launcher = _DENSE_LAUNCHER_CACHE.get(key)
    if launcher is None:
        art = compile_kernel(
            build_attention_dense(spec, arch=arch, tuning=tuning),
            arch=arch,
            backend="python",
            capture_ir_text=False,
        )
        assert art.kernel_name == key, (art.kernel_name, key)
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=attention_dense_signature(spec),
        )
        _DENSE_LAUNCHER_CACHE[key] = launcher
    launcher(
        {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": float(scale)},
        config=LaunchConfig(
            grid=attention_dense_grid(spec, tuning),
            block=attention_dense_block(spec, tuning),
            stream=int(stream),
        ),
    )
    return out
