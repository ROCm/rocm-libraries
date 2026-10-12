# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash Attention for gfx1201 -- **aiw**, the all-in-one kernel.

``aiw`` = "all-in-one". The ``w`` stands in for the ``o`` of "one" on purpose:
``aio`` reads as "async IO" and ``ai1`` reads as "Artificial Intelligence One",
so neither was usable. Please do not "fix" the spelling.

This module unifies three previously separate gfx1201 kernels --
``flash_attn_func_gfx1201.py`` (baseline), ``..._bp.py`` (binding prefetch) and
``..._m32.py`` (two Q row sub-tiles per wave) -- which were never three designs.
They were one design at three points in a knob space, plus drift. Each knob
below is a ``const_expr`` switch resolved at trace time, so a given build emits
exactly one variant's code with no runtime branching.

    knob              baseline      bp             m32
    ---------------------------------------------------------------
    K_PREFETCH_DIST   0             1              1
    V_LDS_LAYOUT      "row"         "transposed"   "transposed"
    ROW_SUBTILES      1             1              2
    QK_SHARDS         1             bp_qk_shards() 1
    VO_CHUNKS         1             vo_chunks()    1
    VO_WIDTH          slice of D    head_dim       head_dim

The three originals are kept on disk unchanged and serve as correctness
oracles: for every knob setting that reproduces one of them, aiw must match it
bitwise wherever the floating-point reduction order is unchanged (see
``test_flash_attn_func_gfx1201_aiw.py``). ISA-level divergence from the
originals is expected and accepted -- aiw uses the 64-bit-base + 32-bit-offset
addressing scheme everywhere, which only ``bp`` had.

--- The knobs -------------------------------------------------------------

``K_PREFETCH_DIST`` -- 0 or 1. gfx1201 has no direct global->LDS copy, so a KV
tile must transit VGPRs and the only latency hiding available is a *binding*
prefetch: issue the global load early, hold it in registers, consume it later.
At distance 0 the load and the LDS store are adjacent, so the latency is fully
exposed (``global_load; s_wait_loadcnt 0x0; ds_store``). At distance 1 both K
and V ride the loop in registers:

    prologue: load K[0], V[0] -> registers
    iteration i (carrying K[i], V[i]):
        store K[i] -> LDS ; barrier ; issue load K[i+1]   <- flies over GEMM1
        GEMM1 (S = K @ Q^T) ; online softmax
        store V[i] -> LDS ; barrier ; issue load V[i+1]   <- flies over GEMM2
        GEMM2 (O += V^T @ P)

Barrier count is unchanged (2/iteration); the cost is one more live register
set. Distance 1 wins from head_dim 48 up; below that the tiles are small enough
that the extra pressure buys nothing.

``V_LDS_LAYOUT`` -- ``"row"`` stages V as V[kv][d]; GEMM2 then needs 8 strided
scalar LDS reads per operand. ``"transposed"`` stages V^T[d][kv] filled with
``global_load_tr_b128``, whose hardware 16x16 transpose delivers each lane its
8 kv-elements contiguously, so GEMM2 reads one vector per operand and the LDS
store stays contiguous instead of becoming a 16-way scatter. Worth +2.7% at
N >= 4096.

``ROW_SUBTILES`` -- Q row sub-tiles owned by each wave. At 2, one K or V operand
feeds two WMMAs instead of one (halving LDS reads per FLOP) and BLOCK_M
doubles, halving the grid and with it K/V global traffic. Costs
``o_accs + q_b_packs + s_accs`` VGPRs, which scales with head_dim: +64 at
head_dim 64 (fine), +112 at 128 (hits the 256-VGPR cap and spills, -27%).

``QK_SHARDS`` -- waves cooperating on one Q row sub-tile, each reducing over its
own head_dim slice in GEMM1 and owning the matching V/O column slice in GEMM2.
Their partial S values are summed through LDS. Lets large head_dim spread its
register cost across waves.

``VO_CHUNKS`` -- V staging passes, so only ``head_dim/VO_CHUNKS`` columns are
LDS-resident at a time. Keeps the padded K+V tile inside the 64 KiB workgroup
cap at large head_dim, at one extra barrier pair per extra pass.

``VO_WIDTH`` / ``D_OFFSET`` -- V/O column *window*, distinct from ``VO_SLICE``
(which is the per-wave share of a window). Attention is column-separable in V:
``O[:, s] = P @ V[:, s]`` and P does not depend on V, so the V/O width can be a
slice of the QK width. This is what keeps ``o_accs`` (VO_WIDTH/2 VGPRs) and the
V LDS tile in budget above head_dim 256, at the cost of repeating GEMM1 and the
K traffic per window.

--- Register layout -------------------------------------------------------

WMMA 16x16x16, wave32:
  - A/B operand: v8f16 per lane (lane16 = row/col, klane*8 = K-offset)
  - C/D result:  v8f32 per lane, element si = C[klane*8+si][lane16]

Shape:  Q/K/V/O are BHSD (batch, num_heads, seq_len, head_dim); the memory
        layout is free so long as D is innermost. See `_strides_of`.
Grid:   (batch * num_q_tiles * num_heads,)
"""

import math as host_math
from dataclasses import fields

from kernels.attention_rdna4 import fmha_abi_gfx1201 as abi
from kernels.attention_rdna4 import fmha_common_gfx1201 as fmha
from kernels.attention_rdna4.fmha_tuning_gfx1201 import (  # noqa: F401
    FmhaInputMetadata,
    FmhaKnobs,
    default_block_m,
    default_block_n,
    default_prefetch_dist,
    q_tiles_per_block,
    qk_shards,
    resolve_knobs,
    resolve_shards,
    vo_chunks,
)
from kernels.common import buffer_ops

# Import rewrite: the only symbol this file takes from `kernels.common.utils`
# is `smax`, which is branch-local and absent from the released tag the build
# clones, so the name aliases the polyfill wholesale and no call site below
# changes. `buffer_ops` is NOT rewritten: `get_element_ptr` is in the released
# tree.
from kernels.attention_rdna4 import flyc_polyfill as common_utils
from kernels.attention_rdna4.philox import Philox

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "flash_attn_func_gfx1201_aiw_kernel"
_LOG2E = host_math.log2(host_math.e)
_LN2 = 0.6931471824645996  # matches AOTriton's literal exactly

# `dtype_to_elem_type` and `_run_compiled` are inlined copies of
# `kernels.common.kernels_common.dtype_to_elem_type` and
# `kernels.common.tensor_shim._run_compiled`. They are duplicated on purpose:
# this directory is a self-contained prototype that must run with the cwd set
# to it and no PYTHONPATH, which puts `kernels.*` out of reach. Fold them back
# into the shared modules if this graduates out of prototype status.


def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its FlyDSL numeric type."""
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', or 'bf16')"
    )


_COMPILED = abi.new_compiled_cache()


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, fmha.llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(fmha.llvm_value(value), fmha.llvm_value(ptr))


# Causal alignment as a *window* sentinel, AOTriton's `WindowValue`. These
# occupy two values a real bound never takes, so `Window_left`/`Window_right`
# stay plain signed integers rather than gaining a discriminant.
# The sentinel values themselves live in `fmha_common_gfx1201`, beside the
# kernel-side code that resolves them; the host only needs to emit them.
_WINDOW_TOPLEFT = fmha.WINDOW_TOPLEFT
_WINDOW_BOTRIGHT = fmha.WINDOW_BOTRIGHT


def build_flash_attn_func_aiw_module_primary(meta, knobs):
    """Build the unified gfx1201 flash-attention kernel.

    Takes the two objects rather than 26 loose parameters, split on *who
    decides*: `problem` is what the caller asked for, `schedule` is what the
    tuning policy answered. Build `schedule` with
    `fmha_tuning_gfx1201.resolve_knobs(problem)` -- every field must be
    resolved, since nothing here falls back to a policy any more.

    `build_flash_attn_func_aiw_module` below is the keyword-argument wrapper
    for callers that only want to name a problem.

    See the module docstring for what each knob selects and which of the three
    original kernels a given setting reproduces.
    """
    # Unpacked to locals so the body below reads unchanged. Anything not
    # resolved is a caller error, not a default to be invented here.
    num_heads = meta.num_heads
    causal = meta.causal
    dtype_str = meta.dtype_str
    sm_scale = meta.sm_scale
    causal_type = meta.causal_type
    bias = meta.bias
    # Modification 3: an attention sink (gpt-oss), a baked axis.
    HAS_SINK = bool(meta.sink)
    # Modification 5: the additive mask's dtype and its key-axis stride, baked.
    BIAS_F32 = bool(meta.bias_f32)
    BIAS_KSTRIDED = bool(meta.bias_key_strided)
    dropout = meta.dropout
    philox_width = meta.philox_width

    WAVES_PER_EU = knobs.waves_per_eu
    FLAT_WORK_GROUP_SIZE = knobs.flat_work_group_size
    BLOCK_M_KNOB = knobs.block_m
    BLOCK_N_KNOB = knobs.block_n
    BLOCK_DMODEL = knobs.block_dmodel
    BLOCK_DMODEL_V = knobs.block_dmodel_v
    D_OFFSET = knobs.d_offset
    K_PREFETCH_DIST_KNOB = knobs.k_prefetch_dist
    V_PREFETCH_DIST = knobs.v_prefetch_dist
    V_LDS_LAYOUT = knobs.v_lds_layout
    STRIDES_CONSTEXPR = knobs.strides_constexpr
    PADDED_HEAD = knobs.padded_head
    ROW_SUBTILES = knobs.row_subtiles
    SHARDS = knobs.shards
    UNSAFE_FP_MATH = knobs.unsafe_fp_math
    FAST_FP_MATH = knobs.fast_fp_math
    DENORMALS_ARE_ZERO = knobs.denormals_are_zero
    SCHED_STRATEGY_KNOB = knobs.sched_strategy
    LPT_TILE_ORDER_KNOB = knobs.lpt_tile_order
    UNSAFE_NO_KV_CLAMP = knobs.unsafe_no_kv_clamp
    KV_ADDR_HOIST = knobs.kv_addr_hoist
    FP_MODE = knobs.fp_mode
    """Build the unified gfx1201 flash-attention kernel.

    See the module docstring for what each knob selects and which of the three
    original kernels a given setting reproduces.
    """

    # ---- WMMA / wave32 constants ----
    #
    # The first four are the hardware: RDNA4 runs wave32 and its WMMA is
    # 16x16x16. The last two are *derived* from those, not free parameters, and
    # are spelled out here because both read as magic at the use site.
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16

    # KV columns spanned by one *pair* of S accumulators -- `2 * WMMA_N`.
    #
    # The pair, rather than the single accumulator, is the unit because it is
    # where the two GEMMs meet. GEMM1 emits `NUM_S_ACCS = COL_SUBTILES * 2`
    # accumulators of `WMMA_N` columns each, and GEMM2 consumes the same span
    # as `PV_K_STEPS = COLS_PER_SUBTILE // WMMA_K` steps of K. Both equal 2, so a
    # sub-tile is exactly one GEMM1 output pair and one GEMM2 input pair.
    #
    # The mapping this implies is open-coded wherever accumulators are indexed
    # by column: accumulator `st` starts at KV column
    # `(st // 2) * COLS_PER_SUBTILE + (st % 2) * WMMA_N`.
    COLS_PER_SUBTILE = 2 * WMMA_N

    # K elements each lane holds of a WMMA A/B operand.
    #
    # A wave32 WMMA spreads `WMMA_M` rows over 32 lanes, so two lanes share
    # each row and split the K extent between them: `WMMA_K / (WARP_SIZE /
    # WMMA_M)` = 8. That is why the operand is a v8f16 and why `klane` (the
    # half-wave index, `lane // 16`) appears multiplied by this everywhere a
    # column is computed -- half-wave 0 covers K 0..7, half-wave 1 covers 8..15.
    # See the operand-layout note in the module docstring.
    WMMA_LANE_K = WMMA_K // (WARP_SIZE // WMMA_M)

    # ---- Knob resolution ----
    K_PREFETCH_DIST = K_PREFETCH_DIST_KNOB
    # K and V prefetch distances are INDEPENDENT. The baseline kernel is
    # (K=0, V=1) -- its "pre-issue first V global load before loop" carries V in
    # registers exactly as bp does, and only K is staged at distance 0. Folding
    # the two into one knob produces a (K=0, V=0) schedule that exists in none
    # of the originals and costs 9.6% at BLOCK_DMODEL 32 non-causal.
    V_LDS_LAYOUT = (
        ("transposed" if K_PREFETCH_DIST else "row")
        if V_LDS_LAYOUT is None
        else V_LDS_LAYOUT
    )
    V_TRANSPOSED = V_LDS_LAYOUT == "transposed"

    ROWS_PER_WAVE = WMMA_M * ROW_SUBTILES

    BLOCK_N = BLOCK_N_KNOB

    COL_SUBTILES = BLOCK_N // COLS_PER_SUBTILE
    NUM_S_ACCS = COL_SUBTILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8

    # V/O column *window*: the slice of the output width this build computes.
    # Distinct from VO_SLICE (a wave's share of a window) and from VO_CHUNK_COLS
    # (a staging pass's share of a window).
    VO_WIDTH = BLOCK_DMODEL_V

    # Head-dimension sharding. QK_SHARDS waves cooperate on one Q row sub-tile,
    # each reducing over its own BLOCK_DMODEL slice in GEMM1 and owning the matching
    # V/O column slice in GEMM2. QK_SHARDS == 1 is the unsharded kernel: every
    # sharded construct below is behind `const_expr(QK_SHARDS > 1)`.
    # Resolved against the V/O *window*, not BLOCK_DMODEL, so a narrow window does
    # not inherit a shard count it cannot divide.
    if SHARDS is not None:
        QK_SHARDS = SHARDS
    elif K_PREFETCH_DIST == 0 or ROW_SUBTILES > 1:
        QK_SHARDS = 1
    else:
        QK_SHARDS = resolve_shards(BLOCK_DMODEL, VO_WIDTH, BLOCK_N)
    QK_SLICE = BLOCK_DMODEL // QK_SHARDS  # head-dim columns per wave in GEMM1

    VO_CHUNKS = vo_chunks(VO_WIDTH, BLOCK_N, QK_SHARDS) if V_TRANSPOSED else 1
    VO_CHUNK_COLS = VO_WIDTH // VO_CHUNKS  # V columns resident per pass
    VO_SLICE = VO_CHUNK_COLS // QK_SHARDS  # V/O columns per wave per pass

    # ---- Validity predicate over the knob space ----
    #
    # Assertions, not ValueError: every name below was resolved by
    # `fmha_tuning_gfx1201.resolve_knobs`, so a violation is that module
    # contradicting itself, not a caller mistake. Caller input is validated
    # with ValueError in `plan()` and in the launcher.
    #
    # Grouped rather than scattered through the derivation so the buildable
    # subset of the knob space can be read in one place.

    # Shapes and enumerations.
    assert (
        BLOCK_DMODEL % 16 == 0 and 16 <= BLOCK_DMODEL <= 512
    ), f"aiw needs 16 <= BLOCK_DMODEL <= 512 and BLOCK_DMODEL % 16 == 0, got {BLOCK_DMODEL}"
    assert dtype_str in ("f16", "bf16"), f"aiw supports f16/bf16, got {dtype_str!r}"
    assert K_PREFETCH_DIST in (
        0,
        1,
    ), f"K_PREFETCH_DIST must be 0 or 1, got {K_PREFETCH_DIST}"
    assert V_PREFETCH_DIST in (
        0,
        1,
    ), f"V_PREFETCH_DIST must be 0 or 1, got {V_PREFETCH_DIST}"
    assert V_LDS_LAYOUT in (
        "row",
        "transposed",
    ), f"V_LDS_LAYOUT must be 'row' or 'transposed', got {V_LDS_LAYOUT!r}"
    assert ROW_SUBTILES in (1, 2), f"ROW_SUBTILES must be 1 or 2, got {ROW_SUBTILES}"

    # BLOCK_N. The power of two is load-bearing: `_sdiv_rd` in the kernel is an
    # arithmetic shift, which is only a floor-division when BLOCK_N is one.
    assert (
        BLOCK_N % COLS_PER_SUBTILE == 0
    ), f"BLOCK_N ({BLOCK_N}) must be a multiple of COLS_PER_SUBTILE ({COLS_PER_SUBTILE})"
    assert BLOCK_N & (BLOCK_N - 1) == 0, f"BLOCK_N ({BLOCK_N}) must be a power of two"

    # The V/output window, and how it divides.
    assert (
        VO_WIDTH % 16 == 0 and 0 < VO_WIDTH <= BLOCK_DMODEL
    ), f"BLOCK_DMODEL_V must be a positive multiple of 16 and <= BLOCK_DMODEL, got {VO_WIDTH}"
    assert (
        D_OFFSET % 16 == 0 and D_OFFSET + VO_WIDTH <= BLOCK_DMODEL
    ), f"D_OFFSET {D_OFFSET} + BLOCK_DMODEL_V {VO_WIDTH} must fit in BLOCK_DMODEL {BLOCK_DMODEL}"
    assert (
        VO_SLICE % WMMA_N == 0
    ), f"V/O slice {VO_SLICE} must be a multiple of WMMA_N={WMMA_N}"

    # Sharding. A slice not a multiple of WMMA_K would silently drop part of the
    # reduction: BLOCK_DMODEL 224 with 4 shards gives a 56-wide slice, of which
    # only 48 would be reduced (measured rel err 0.97).
    assert BLOCK_DMODEL % QK_SHARDS == 0 and QK_SLICE % WMMA_K == 0, (
        f"BLOCK_DMODEL {BLOCK_DMODEL} with {QK_SHARDS} SHARDS gives a {QK_SLICE}-wide "
        f"slice, which must be a multiple of WMMA_K={WMMA_K}"
    )

    # Combinations the kernel does not implement. Written as conditionals
    # rather than negated conjunctions -- `not (not V_TRANSPOSED and ...)` is
    # a sentence nobody can read twice the same way.
    if not V_TRANSPOSED:
        assert (
            QK_SHARDS == 1
        ), "V_LDS_LAYOUT='row' does not implement cross-shard reduction; use 'transposed'"
        assert (
            VO_CHUNKS == 1
        ), "V_LDS_LAYOUT='row' does not implement chunked V staging; use 'transposed'"
    if VO_CHUNKS > 1:
        assert V_PREFETCH_DIST, "chunked V staging requires V_PREFETCH_DIST=1"
    if ROW_SUBTILES > 1:
        assert (
            QK_SHARDS == 1
        ), "ROW_SUBTILES > 1 with qk_shards > 1 is untested; pick one"
    if causal:
        # The causal mask indexes s_accs as a flat 16, which an unrolled loop
        # over a longer list walks off the end of -- an IndexError at trace
        # time rather than a wrong answer. (Dies with the interval work.)
        assert NUM_S_VALS == 16, (
            f"causal masking requires BLOCK_N == {COLS_PER_SUBTILE} (NUM_S_VALS == 16), "
            f"got BLOCK_N={BLOCK_N} (NUM_S_VALS={NUM_S_VALS})"
        )
    # These combinations are not implemented rather than not expressible. Fail
    # at build time; do not emit a kernel that silently computes the wrong
    # thing.

    # ---- Workgroup geometry ----
    if K_PREFETCH_DIST == 0:
        BLOCK_M = BLOCK_M_KNOB
        # Stays here rather than in the section above: BLOCK_M only exists on
        # this branch, and on the other one it is derived rather than given.
        assert (
            BLOCK_M % ROWS_PER_WAVE == 0
        ), f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}"
        Q_TILES_PER_BLOCK = BLOCK_M // ROWS_PER_WAVE
        NUM_WAVES = Q_TILES_PER_BLOCK
    else:
        # Keep the workgroup at TARGET_WAVES by trading Q row sub-tiles for SHARDS,
        # so BLOCK_M shrinks as QK_SHARDS grows.
        #
        # Divide by ROW_SUBTILES so BLOCK_M is *invariant* to that knob and only
        # the wave count changes: at ROW_SUBTILES=2 the same rows are covered by
        # half as many waves, each doing twice the work, which is the whole
        # point (one K/V operand feeds two WMMAs). Without the division a
        # ROW_SUBTILES=2 build would silently double BLOCK_M as well, doubling
        # per-wave register pressure on top of the knob's own cost.
        #
        # Note this is invisible to a bitwise output comparison -- each Q row's
        # arithmetic is identical however rows are grouped into blocks -- so
        # only the benchmark catches it.
        Q_TILES_PER_BLOCK = max(
            1, q_tiles_per_block(BLOCK_DMODEL, QK_SHARDS) // ROW_SUBTILES
        )
        BLOCK_M = ROWS_PER_WAVE * Q_TILES_PER_BLOCK
        NUM_WAVES = Q_TILES_PER_BLOCK * QK_SHARDS

    if FLAT_WORK_GROUP_SIZE is None:
        FLAT_WORK_GROUP_SIZE = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = FLAT_WORK_GROUP_SIZE

    BLOCK_N_OUT = BLOCK_N

    # LLVM's amdgpu-sched-strategy function attribute; "" leaves the default
    # GCN scheduler in place. See the passthrough block in the launch wrapper.
    # Measured at BATCH=2 H=12 N=4096 d=128 f16 -- distance 1: causal
    # 85.6 -> 88.5 TFLOPS, non-causal 91.4 -> 91.9. Distance 0: causal
    # 69.8 -> 79.2, but non-causal 89.4 -> 88.6, so only causal wants it there.
    # `None` means the policy below; pass `""` for the stock GCN scheduler.
    SCHED_STRATEGY = (
        ("max-memory-clause" if (K_PREFETCH_DIST or causal) else "")
        if SCHED_STRATEGY_KNOB is None
        else SCHED_STRATEGY_KNOB
    )

    K_STEP_QK = WMMA_K
    K_STEPS_QK = QK_SLICE // K_STEP_QK  # GEMM1 K-steps for this wave's slice

    D_CHUNK = WMMA_N
    D_CHUNKS = VO_SLICE // D_CHUNK  # accs per wave per chunk
    O_ACCS = VO_CHUNKS * D_CHUNKS  # accs live across the KV loop, per Q tile

    PV_K_STEP = WMMA_K
    PV_K_STEPS = COLS_PER_SUBTILE // PV_K_STEP

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(BLOCK_DMODEL)

    NUM_HEADS = num_heads
    HEAD_DIM = BLOCK_DMODEL
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # Strides and sm_scale are runtime arguments, not folded constants: an AOT
    # kernel cannot bake them in, since a fixed set of binaries has to cover
    # every shape.
    #
    # Measured price (B=1 H=8 N=4096 f16, interleaved 3-rep A/B over the full
    # BLOCK_DMODEL ladder x causal): **median ratio 0.996**, worst 0.967 (BLOCK_DMODEL
    # 16 causal), best 1.041 (BLOCK_DMODEL 192 causal). Several configs come out
    # *faster* and the spread is symmetric about 1.0, so this is the board's
    # noise floor rather than a measurable cost. Registers: +0 to +4 VGPRs,
    # +22 SGPRs, no new spills at any BLOCK_DMODEL. Output is bitwise identical to
    # the folded form, sm_scale included.
    #
    # Each tensor carries its own triple, and they are not interchangeable: K
    # and V reach the kernel exactly as the caller allocated them (`mha_fwd_aot`
    # passes them through untouched), and under MQA/GQA they carry Num_head_k
    # rather than Num_head_q. Going from one shared triple to four cost +18
    # SGPRs and zero VGPRs -- strides are uniform scalars, so they only change
    # which value an address multiplies.
    #
    # `STRIDES_CONSTEXPR=True` keeps them folded. It is retained only as an A/B
    # arm for future phases -- if addressing ever becomes expensive we want to
    # be able to measure against the folded form -- and is not a shipping
    # configuration.
    #
    # Strides arrive in **BHSD slot order**: batch, head, sequence. Axis 3 is
    # `D`, contiguous by contract, so it is never passed.
    #
    # Named for the axis, not the slot. An earlier version used `stride_q0/1/2`
    # deliberately, because the letters inherited from the maths
    # (`stride_qz/qh/qm`) read badly and caused real mix-ups during AOTriton's
    # kernel development -- but that objection was to *cryptic* letters, and
    # numeric slots trade one unreadable convention for another. Nothing at
    # runtime distinguishes a head stride from a sequence stride, so a caller
    # that swaps them gets finite garbage rather than an error; spelling the
    # axis out is the only check there is.
    #
    # Bias already used this order (`stride_b_batch/1/2` are batch, head, Sq), so
    # this makes the five tensors agree rather than introducing a convention.
    #
    # Longest-processing-time-first dispatch for causal. Under causal masking
    # a workgroup's cost grows with its q_tile (tile 0 walks one KV block, tile
    # N-1 walks N), and grid.y is dispatched in increasing order -- so the
    # cheapest blocks go first and the most expensive land in the tail, which
    # is the worst possible order. Reversing the index puts the expensive
    # blocks first and leaves only cheap ones to fill the tail.
    #
    # Measured B=1 H=8 N=4096 f16 causal, TFLOPS forward -> reversed:
    #   BLOCK_DMODEL  16   31.8 -> 35.7  (+12%)
    #             32   51.2 -> 59.5  (+16%)
    #             64   67.1 -> 77.8  (+16%)
    #            128   74.8 -> 87.0  (+16%)
    #            256   68.8 -> 72.8  (+6%)
    #            512   43.9 -> 46.6  (+6%)
    #
    # Non-causal is untouched: every tile costs the same there, so the reversal
    # would be pure arithmetic for no gain and is not emitted.
    #
    # This is orthogonal to the *axis* order (see the grid comment below) --
    # that one decides whether a scheduling group has uniform duration, this
    # one decides the order the groups are issued in. Both matter, and neither
    # of the pre-unification kernels had either.
    # Longest-processing-time-first dispatch of the Q tiles. Named for what it
    # is for rather than what it does: under causal masking a tile's cost grows
    # with its index, so issuing the expensive ones first leaves only cheap
    # tiles to fill the tail. With uniform cost -- every non-causal tile -- the
    # reversal is a permutation with no load-balancing content, so `and CAUSAL`
    # is part of the knob's *definition* and is resolved here rather than
    # re-tested at the use site, where it read as an arbitrary restriction on
    # an unrelated flag.
    _LPT_TILE_ORDER = CAUSAL and LPT_TILE_ORDER_KNOB
    # Measurement-only: drops the KV row clamp. UNSAFE in general -- it is
    # what buffer bounds checking would replace -- and now the *only* way the
    # clamp comes off, so read the precondition carefully.
    #
    # The condition is `seqlen_k % BLOCK_N == 0`, not the "seq_len a multiple of
    # BLOCK_M" this said before. BLOCK_M bounds the Q axis and has nothing to
    # do with it: what over-reads is the ragged KV *tail* tile, whose rows run
    # to `ceil(seqlen_k / BLOCK_N) * BLOCK_N`. Nothing checks either, at build
    # time or at launch, so a benchmark that turns this on and then changes its
    # sequence length reads off the end of K and V.
    _NO_KV_CLAMP = UNSAFE_NO_KV_CLAMP
    # Floating-point latitude granted to the compiler.
    #
    # "noninf" (default) is `fast` minus `ninf`, and drops the function-level
    # `unsafe-fp-math` / `no-nans-fp-math` attributes. `denormal-fp-math-f32`
    # (DAZ) is kept in every mode -- it is about denormals, not infinities, and
    # it is where the actual win is.
    #
    # Why: `ninf` lets the compiler assume no operand is infinite, so an -inf
    # flowing through a fast-math op may simply be folded away. That is not
    # hypothetical -- it silently deleted the KV tail mask (see the comment
    # there), and it will do the same to a bias tensor, where a boolean
    # attention mask cast to float is exactly a matrix of -inf.
    #
    # Cost of giving it up, measured with DAZ held constant (B=1 H=8 N=4096
    # f16, BLOCK_DMODEL 16/64/128/256/512 x causal): **within noise everywhere** --
    # 91.5 vs 91.9 TFLOPS at BLOCK_DMODEL 128 non-causal, 45.5 vs 45.4 at 512. The
    # permission bought nothing and cost a silent miscompile.
    #
    # `nnan` is retained. NaN can only arise here from -inf minus -inf, which
    # the m_i floor rules out, and the API contract excludes
    # NaN inputs.
    #
    # "fast" restores the old behaviour for A/B; "safe" additionally drops
    # `nnan` (~0.6%).
    # Host-side: `fp_mode` is const_expr, so this is a Python object the kernel
    # body captures rather than a traced value.
    fastmath = fmha.FastMath(FP_MODE)

    # Two softmax corrections, unconditional. Both come from AOTriton's
    # hard-won list and there is no reason to keep the un-corrected form
    # reachable -- a knob selecting known-wrong numerics is a liability, not a
    # feature.
    #
    # (a) `m_i` initialises to -3.40282e+38, not -inf. If a tile is entirely
    #     masked its row max is -inf, and with an -inf init the rescale becomes
    #     exp2(-inf - -inf) = exp2(NaN) = NaN. A finite floor makes it
    #     exp2(0) = 1 and the masked probabilities exp2(-inf - m) = 0, which is
    #     the right answer. The *mask* fill stays -inf; only the init changes.
    #
    # (b) The QK scale is applied to the scores **before** the row max, so
    #     `m_i` lives in the scaled domain and the exponent is a plain
    #     subtract. The alternative -- `exp2(fma(s, qk_scale, -qk_scale*m))` --
    #     is exactly the FMA pattern AOTriton flags in ROCm/aotriton#54, and it
    #     measurably loses accuracy at large input magnitudes: at BLOCK_DMODEL 128
    #     causal the corrected form is *exact* against an fp64 reference from
    #     magnitude 300 up, where the FMA form sits at 4-7e-4.

    # `BLOCK_DMODEL` is BLOCK_DMODEL: the compile-time tile width, drawn from the
    # ladder. The *real* extents are the runtime `hdim_qk` / `hdim_vo`
    # arguments, and PADDED_HEAD says whether they differ from the tile.
    # AOTriton derives exactly this (`attn_fwd.cc`):
    #     hdim_rounded = round_value(max(hdim_qk, hdim_vo), ladder)
    #     PADDED_HEAD  = (hdim_rounded != hdim_qk || hdim_rounded != hdim_vo)
    # With PADDED_HEAD false the two are equal to the tile and no masking is
    # emitted at all -- that is the common case and the one the ladder measures.

    # Causal masking. `causal_type` is the *caller's* vocabulary, AOTriton's
    # CAUSAL_TYPE: 0 = none, 1 = top-left aligned, 2 = bottom-right aligned,
    # 3 = an explicit sliding window. The two alignments differ only in where
    # the diagonal sits, and they coincide when seqlen_q == seqlen_k -- which
    # is why a single `causal` bool sufficed for a long time.
    #
    #   top-left      key j is visible to query i iff  j <= i
    #   bottom-right  ...                        iff  j <= i + (seqlen_k - seqlen_q)
    #
    # 3 = generalized sliding window: the test becomes a two-sided band,
    # `i - window_left <= j <= i + window_right`, with both bounds signed
    # runtime arguments. It subsumes 1 and 2 exactly -- (seqlen_q, 0) and
    # (seqlen_q, seqlen_k - seqlen_q) respectively -- which is why AOTriton
    # ships only {0, 3} and resolves 1/2 on the host. Types 1 and 2 survive
    # here only until that equivalence is nailed down by a test; see
    # sdpa-gswa-plan.md.
    #
    # PyTorch's is_causal=True is top-left; see
    # https://github.com/pytorch/pytorch/issues/108108 for the debate about
    # changing that default.
    if causal_type is None:
        HOST_CAUSAL_TYPE = 1 if causal else 0
    else:
        HOST_CAUSAL_TYPE = causal_type
    if HOST_CAUSAL_TYPE not in (0, 1, 2, 3):
        raise ValueError(f"causal_type must be 0, 1, 2 or 3, got {HOST_CAUSAL_TYPE}")
    if bool(HOST_CAUSAL_TYPE) != bool(causal):
        raise ValueError(
            f"causal={causal} disagrees with causal_type={HOST_CAUSAL_TYPE}"
        )
    # **The kernel only ever sees 0 or 3.** 1 and 2 are host-side conveniences
    # that resolve to a window before dispatch, which is what AOTriton ships
    # (`@ati.scalar('CAUSAL_TYPE', options=[0, 3])`). Keeping them in the
    # kernel would leave two ways to express one diagonal, free to drift apart
    # under maintenance. The window path reproduces both *bitwise*, which is
    # what licensed removing them; see sdpa-gswa-plan.md section 0.
    CAUSAL_TYPE = 0 if HOST_CAUSAL_TYPE == 0 else 3

    # Bias tensor, AOTriton's BIAS_TYPE. 0 = none, 1 = a (B, H, Sq, Sk) matrix
    # added to the scores before the softmax. A build axis, so BIAS_TYPE == 0
    # emits nothing at all -- the loop is latency-bound and a bias that costs
    # anything when unused would be paid by every caller who does not want it.
    BIAS_TYPE = 1 if bias else 0
    # Modification 4 (hipDNN vendoring): bias *and* a causal mask or window.
    # Upstream refuses the pair as undefined, following PyTorch. hipDNN and cuDNN
    # define it: the bias is added and the mask still removes the keys it
    # removes. That is what this kernel computes once the refusal is gone -- in
    # every tile the order is scale, bias, mask, and the mask writes -inf
    # whatever the bias held -- so nothing else changes.

    # Dropout, AOTriton's ENABLE_DROPOUT. A build axis, so a build without it
    # emits no PRNG at all -- the loop is latency-bound and a caller who does
    # not want dropout should not pay for the option.
    #
    # The PRNG itself lives in `philox.py`: it is not attention, the backward
    # pass and the debug mask kernel need the identical stream, and this file
    # is long enough. What stays here is the *offset scheme* -- which element
    # gets which offset -- because that is layout-specific.
    ENABLE_DROPOUT = bool(dropout)
    PHILOX = Philox.for_arch() if philox_width is None else Philox(width=philox_width)

    # KV columns past seqlen_k are always masked: seqlen need not divide
    # BLOCK_N now that the interface no longer pads. The guard is dynamic, so
    # interior tiles cost one scalar compare.

    # ---- LDS layout ----
    # K is padded rather than XOR-swizzled (a swizzle was implemented and
    # measured a net loss; see sdpa_lore_gfx1201.md). Chunking bounds the V
    # window, so the padding always fits.
    _LDS_PAD = 4
    K_STRIDE = HEAD_DIM + _LDS_PAD
    # Transposed V: V^T[d][kv]. +4 makes the row stride 36 elems = 72 B, i.e.
    # 18 dwords, so lane16*18 mod 32 hits 16 distinct banks (conflict-free)
    # while staying 8-byte aligned.
    VT_STRIDE = BLOCK_N + _LDS_PAD
    V_STRIDE = VO_WIDTH + _LDS_PAD  # row-major V

    # Cooperative-load vector width, in elements. 8 == 16 bytes.
    #
    # Fixed at 8, which is exactly what the alignment contract guarantees: the
    # D-axis pitch is a multiple of 16 bytes, nothing more. A 16-element
    # (32-byte) load needs 32-byte alignment, and there is no way to establish
    # it -- the row address is `base + row * stride_seq`, and `stride_seq` need
    # only be a multiple of the pitch. A tensor whose pitch is an odd multiple
    # of 8 elements (say a 16-wide head sliced out of a 24-wide allocation)
    # puts every odd row on a 16-byte boundary, and the wider load is then
    # undefined behaviour. This is the same over-promised-alignment failure
    # documented on `fmha.lds_load_v8` for LDS, where it cost 2.2x.
    #
    # It is also not a win. Measured 8 against 16 (B=1 H=8 N=4096 f16, TFLOPS):
    #
    #   BLOCK_DMODEL   non-causal        causal
    #       16    37.4 -> 40.2   28.2 -> 35.7
    #       32    64.0 -> 61.4   47.0 -> 47.2
    #       64    81.0 -> 81.4   74.3 -> 70.2
    #      128    92.3 -> 92.1   83.9 -> 81.9
    #      192    97.7 -> 94.5   74.0 -> 76.5
    #      256    89.1 -> 90.1   72.6 -> 73.7
    #      512    45.7 -> 55.1   43.1 -> 51.6
    #
    # Median +0.8%, best +27% (BLOCK_DMODEL 16 causal), worst -5.5%. So the wider
    # load was buying nothing on average while carrying an alignment hazard
    # that no test would catch -- it would fault or corrupt only on layouts we
    # do not currently generate. Removed rather than tuned: tuning an unsound
    # knob just spreads the hazard across more configs.
    VEC_WIDTH = 8

    def _load_geom(width):
        return fmha.load_geom(width, VEC_WIDTH, BLOCK_SIZE, BLOCK_N)

    # Cover BLOCK_N rows with ceil() batches, not floor(). Flooring silently
    # dropped rows whenever K_ROWS_PER_BATCH neither reached BLOCK_N nor
    # divided it: BLOCK_DMODEL 160/192/224 give 25/21/18, so BLOCK_N // that == 1
    # and only 25/21/18 of the 32 KV rows reached LDS. The rest was stale LDS,
    # which surfaced as NaN.
    K_TPR_LOAD, K_ROWS_PER_BATCH, NUM_BATCHES_K, K_NEEDS_GUARD = _load_geom(HEAD_DIM)

    # global_load_tr_b128 transposes an 8x8 tile of 16-bit elements across each
    # group of 8 lanes, so one wave-wide TR load produces a 16(d) x 16(kv) block
    # already in WMMA-operand layout. Split those blocks over the waves.
    V_TR_D_BLOCKS = VO_CHUNK_COLS // WMMA_N
    _V_TR_TILES = V_TR_D_BLOCKS * (BLOCK_N // WMMA_K)
    # The V TR tiling need not divide evenly across the waves: tail tiles are
    # guarded at the LDS store. Requiring divisibility used to force BLOCK_DMODEL
    # 160 down to 4 waves, which cost it 89.1 -> 70.0 TFLOPS.
    V_TR_LOADS = (_V_TR_TILES + NUM_WAVES - 1) // NUM_WAVES
    V_TR_NEEDS_GUARD = V_TR_LOADS * NUM_WAVES != _V_TR_TILES

    V_TPR_LOAD, V_ROWS_PER_BATCH, NUM_BATCHES_V, V_NEEDS_GUARD = _load_geom(
        VO_CHUNK_COLS
    )

    # How many register-resident V vectors ride the loop, under either layout.
    V_LOADS = V_TR_LOADS if V_TRANSPOSED else NUM_BATCHES_V

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = VO_CHUNK_COLS * VT_STRIDE if V_TRANSPOSED else BLOCK_N * V_STRIDE
    LDS_K_TOTAL_SIZE = LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    # The cross-shard S reduction aliases the V region rather than allocating
    # its own: V is written to LDS only *after* softmax, so between the
    # post-K-store barrier and that write the V tile holds the previous
    # iteration's data, already consumed by the previous GEMM2.
    RED_F32_PER_WAVE = NUM_S_VALS * WARP_SIZE
    RED_F32_TOTAL = NUM_WAVES * RED_F32_PER_WAVE
    RED_ALIASES_V = QK_SHARDS == 1 or RED_F32_TOTAL * 4 <= LDS_V_TOTAL_SIZE * 2
    if not RED_ALIASES_V:
        LDS_KV_TOTAL_SIZE += (RED_F32_TOTAL * 4 + 1) // 2  # in elem_dtype units

    # FlyDSL's `dtype_to_elem_type` returns a Numeric class, which is what the
    # Vector API (`Vec.make_type`, `.to(...)`) and `fx.Array` require.
    elem_numeric_cls = dtype_to_elem_type(dtype_str)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem_numeric_cls, LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_aiw_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
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
        philox_seed_output: fx.Pointer,
        philox_offset_output: fx.Pointer,
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
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        SINK: fx.Pointer,
        sink_stride: fx.Int64,
        stride_b_seq_k: fx.Int64,
    ):
        elem_type = elem_numeric_cls.ir_type
        elem_dtype = elem_numeric_cls

        def _to_global_ptr_i64(ptr):
            return fx.as_ir_value(fx.Int64(fx.ptrtoint(ptr)))

        q_ptr = fmha.pointer_to_llvm_ptr(Q)
        k_ptr = fmha.pointer_to_llvm_ptr(K)
        v_ptr = fmha.pointer_to_llvm_ptr(V)
        v_ptr_i64 = _to_global_ptr_i64(V)
        o_ptr = fmha.pointer_to_llvm_ptr(O)
        v8f16_type = Vec.make_type(8, elem_dtype)
        vxf16_type = Vec.make_type(VEC_WIDTH, elem_dtype)

        # ---- Varlen prologue: VarlenBits -> six scalars ----
        #
        # The *only* place the layout is examined. Everything downstream reads
        # the scalars and cannot tell which of the configurations in
        # sdpa-varlen-plan.md section 2 it is running under -- which is that
        # plan's objective 3, and why there is no `if varlen_mode` in the body.
        #
        # `z` is uniform across the workgroup, so every load here is scalar:
        # at most four, once, into SGPRs. They do not touch the VGPR budget.
        #
        # Real branches, not selects. A select-based decode would issue the
        # loads unconditionally and fault on the null `seqinfo` pointers that
        # the dense case passes; the branch also keeps `VarlenBits == 0` free
        # rather than merely correct.
        z_i32 = fx.Int32(gpu.block_idx.z)

        seqlen_q_i32, q_row_off, q_batch = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32
        )
        seqlen_k_i32, k_row_off, k_batch = fmha.decode_addressing(
            varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32
        )
        lse_tokens = fmha.lse_token_pitch(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens
        )

        seqlen_k_v = fx.Index(seqlen_k_i32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_kv = lds.kv.ptr

        # f32 view of the V LDS region, for the cross-shard S reduction. The kv
        # array is elem_dtype (16-bit), so go through an addrspace(3) LLVM
        # pointer: ptrtoint on a shared pointer yields the 32-bit LDS offset.
        _lds_byte_base = fx.as_ir_value(fx.ptrtoint(lds_kv))
        _RED_BYTE0 = (
            LDS_V_BASE
            if RED_ALIASES_V
            else LDS_KV_TOTAL_SIZE - (RED_F32_TOTAL * 4 + 1) // 2
        ) * 2

        tid, wave_id, lane, lane16, klane = fmha.wave_lanes(WARP_SIZE)

        # (q_tile, shard) decomposition of the wave index. At QK_SHARDS == 1
        # this is q_tile == wave_id and shard == 0, i.e. the unsharded mapping.
        q_tile_in_block = wave_id // QK_SHARDS
        shard_id = wave_id % QK_SHARDS
        wave_q_offset = q_tile_in_block * ROWS_PER_WAVE

        # Column origins of this wave's slices. Both are 0 at QK_SHARDS == 1.
        shard_qk_off = shard_id * fx.Index(QK_SLICE)  # into Q/K BLOCK_DMODEL
        shard_vo_off = shard_id * fx.Index(VO_SLICE)  # into the V/O window

        # 3D grid: (head_q, q_tile, batch). A flat grid would need two integer
        # divisions here to recover these, and with num_heads runtime neither
        # would fold away.
        #
        # **The axis order is load-bearing for causal.** Under causal masking a
        # workgroup's cost grows with its q_tile -- tile 0 walks one KV block,
        # tile N-1 walks N. The x axis dispatches fastest, so putting q_tile
        # there spreads durations 1..N across every scheduling group, while
        # putting head there gives each group a uniform duration. Measured at
        # B=1 H=8 N=4096 f16 causal, q_tile-fastest against head-fastest:
        # BLOCK_DMODEL 16 0.587, 32 0.612, 64 0.715, 128 0.769. Non-causal is
        # indifferent (all within 1%), which is what identifies the cause.
        #
        # Note AOTriton uses dim3{S,H,B} -- q_tile fastest -- for NUM_XCDS == 1.
        # That is not a contradiction: it also forces PERSISTENT_TYPE = 2 for
        # every causal functional, which replaces the grid with a work-stealing
        # loop and makes the axis order irrelevant. Porting its grid_calculator
        # verbatim without persistent-dynamic would reintroduce the regression
        # above. Revisit this ordering when persistent-dynamic lands.
        head_q = fx.Index(gpu.block_idx.x)
        if const_expr(_LPT_TILE_ORDER):
            # Max_seqlen_q, not this sequence's length: the reversal has to
            # be a permutation of the *grid*, whose y extent the host sized
            # from Max_seqlen_q.
            _ntiles = (fx.Index(max_seqlen_q) + (BLOCK_M - 1)) // BLOCK_M
            q_tile_idx = _ntiles - fx.Index(1) - fx.Index(gpu.block_idx.y)
        else:
            q_tile_idx = fx.Index(gpu.block_idx.y)
        start_q = q_tile_idx * BLOCK_M

        # Does this workgroup own any real Q row? Under varlen the grid's Q
        # extent is sized from Max_seqlen_q, so whole workgroups land past the
        # end of a shorter sequence and this is false for them.
        #
        # Compared in i32, and the Q sequence length is never widened past it.
        # Both operands are bounded by Max_seqlen_q, an i32 ABI argument, so 64
        # bits buys nothing -- and it costs: `fx.Index` is *unsigned*, so the
        # widened form lowered to `v_cmp_gt_u64` and every such comparison in
        # this file had to be hand-written as an explicit `arith.cmpi(slt, ...)`
        # to get the signed predicate back. `fx.Int32` is signed, so `<` is
        # already the right thing.
        _alive = fx.Int32(start_q) < seqlen_q_i32

        # **The Q base must be clamped, not just the row within the tile.**
        # `q_tbase(start_q)` folds start_q into the 64-bit base, and the
        # in-bounds guard below only clamps the row *inside* the tile -- so a
        # dead workgroup still addresses `row_off + start_q` rows in, which
        # for a packed tensor runs past the end of the whole allocation, not
        # merely past this sequence. Dense never reached it: there the grid is
        # exactly ceil(seqlen_q / BLOCK_M), so start_q < seqlen_q always.
        #
        # Faults for real -- a 1.3 MB overshoot on a 16-sequence batch hits an
        # unmapped page. It is a *read*, and one whose result is discarded, so
        # smaller overshoots land inside the allocation and are silently
        # harmless, which is exactly why the varlen tests did not catch it.
        _q_start_addr = fx.Index(start_q if _alive else fx.Index(0))

        # MQA/GQA: Num_head_q / Num_head_k query heads share each KV head.
        # The ratio is uniform and computed once, so the scalar divide is
        # immaterial; the per-head division below is by that ratio.
        head_k = head_q // (fx.Index(num_head_q) // fx.Index(num_head_k))

        load_row_in_batch = tid // K_TPR_LOAD
        load_lane_in_row = tid % K_TPR_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        v_row_in_batch = tid // V_TPR_LOAD
        v_col_base = (tid % V_TPR_LOAD) * VEC_WIDTH

        # `max(seqlen_k - 1, 0)`. `fx.Index` is **unsigned**, so a bare
        # `seqlen_k - 1` wraps to 2**64-1 at seqlen_k == 0 and the KV clamp
        # below then pins every address to that row -- the fault lands at
        # 0xfffffffff000, which is that value truncated to the virtual address
        # width.
        #
        # Now unreachable: the only caller that could arrive here with
        # seqlen_k == 0 was the prologue prefetch, and that is skipped when
        # there are no KV tiles. Kept because the hazard is an unsigned wrap,
        # which produces a plausible address rather than an obvious failure,
        # and it costs one scalar op to pin.
        _slast_i32 = common_utils.smax(seqlen_k_i32 - fx.Int32(1), fx.Int32(0))
        seq_last = fx.Index(_slast_i32)

        # ---- Address split: 64-bit uniform base + 32-bit divergent offset ----
        #
        # The full linear element index is
        #     ((batch * seq_len) + token) * nheads * BLOCK_DMODEL + head * BLOCK_DMODEL + d
        # i.e. it spans all of B, S, H and D, and overflows i32 at 2G elements
        # (2 GB at f16, which real shapes reach). Only the *intra-tile* part is
        # safely 32-bit: it is bounded by
        #     max(BLOCK_M, BLOCK_N) * nheads * BLOCK_DMODEL + BLOCK_DMODEL
        # because the row index is relative to the tile.
        #
        # So the batch/head/tile origin stays in 64 bits, and it is uniform
        # across the wave -- which is also exactly the shape LLVM's
        # SelectGlobalSAddr folds into an SGPR base plus a 32-bit VGPR offset.
        # Strides, either folded or taken from arguments. The *body* below is
        # identical either way -- FlyDSL's arithmetic accepts a Python int and
        # an fx value interchangeably -- so only this binding differs. That is
        # the whole reason a single kernel source can serve both the JIT and AOT
        # paths (see D2 in sdpa-close-gap-plan1.md).
        if const_expr(STRIDES_CONSTEXPR):
            # Axis 0 (batch) still depends on the runtime seq_len, so it is
            # never a compile-time constant even here; only 1 and 2 fold.
            # Diagnostic arm only, and valid solely when seqlen_q == seqlen_k.
            # Dense-only: it derives the layout from the shape, which varlen
            # invalidates outright. The host rejects the combination.
            # BHSD compact: (batch, head, seq).
            _stq = (
                fx.Index(max_seqlen_q) * STRIDE_TOKEN,
                fx.Index(max_seqlen_q) * HEAD_DIM,
                HEAD_DIM,
            )
            _stk = (
                fx.Index(max_seqlen_k) * STRIDE_TOKEN,
                fx.Index(max_seqlen_k) * HEAD_DIM,
                HEAD_DIM,
            )
            q_st = o_st = _stq
            k_st = v_st = _stk
            sm_log2e = fx.Float32(sm_scale * _LOG2E)
        else:
            q_st = (
                fx.Index(stride_q_batch),
                fx.Index(stride_q_head),
                fx.Index(stride_q_seq),
            )
            k_st = (
                fx.Index(stride_k_batch),
                fx.Index(stride_k_head),
                fx.Index(stride_k_seq),
            )
            v_st = (
                fx.Index(stride_v_batch),
                fx.Index(stride_v_head),
                fx.Index(stride_v_seq),
            )
            o_st = (
                fx.Index(stride_o_batch),
                fx.Index(stride_o_head),
                fx.Index(stride_o_seq),
            )
            sm_log2e = fastmath.mul(sm_scale, fx.Float32(_LOG2E))

        # Q and O are indexed by the query head; K and V by the KV head they
        # share. At num_head_q == num_head_k these coincide.
        _q_batch_v = fx.Index(q_batch)
        _k_batch_v = fx.Index(k_batch)
        _q_row_off_v = fx.Index(q_row_off)
        _k_row_off_v = fx.Index(k_row_off)
        # Bias is (B, H, Sq, Sk): the last axis is the KV column, so unlike
        # Q/K/V/O its "row" stride is stride_b_seq_q and the contiguous axis is the
        # one the KV tile walks. Indexed with the *same* offsets the varlen
        # decode produced, so it inherits every layout for free rather than
        # needing its own -- sdpa-bias-plan.md 3.
        #
        # **Both** offsets: `q_row_off` on the row and `k_row_off` on the
        # column. The column half was missing, so under a packed layout every
        # sequence read sequence 0's bias columns -- a wrong answer rather than
        # an out-of-bounds one, since a KV column is always below `seqlen_k`
        # and so inside the tensor, which is why nothing caught it. It also
        # contradicted the sentence above: the bias inherited the Q decode and
        # ignored the K one.
        #
        # Inert everywhere with coverage today. `k_row_off` is 0 for dense and
        # for the padded varlen layouts; only the packed layouts move, and
        # bias-with-varlen has no test in any of the three suites. That gap is
        # worth naming rather than papering over: this makes the kernel
        # self-consistent, it does not settle what a packed bias should mean.
        if const_expr(BIAS_TYPE):
            _b_ptr = fmha.pointer_to_llvm_ptr(B)
            _b_base = (
                _q_batch_v * fx.Index(stride_b_batch)
                + head_q * fx.Index(stride_b_head)
                + _q_row_off_v * fx.Index(stride_b_seq_q)
            )

        if const_expr(ENABLE_DROPOUT):
            # The offset scheme, and the *only* dropout-specific arithmetic in
            # this file -- everything else is `philox.py`.
            #
            #   stride = cdiv(Max_seqlen_k, RN)
            #   base   = (*offset1 + offset2) + off_zh * Max_seqlen_q * stride
            #   offset(m, n) = base + m * stride + n // RN
            #
            # `BLOCK_M` and `BLOCK_N` appear nowhere: `m` and `n` are global
            # element coordinates, so the mask does not move when the kernel is
            # re-tuned. That is the reproducibility contract of
            # sdpa-dropout-plan.md §3, and it is invisible in any test that
            # uses one tile size.
            #
            # The offset scheme itself is `Philox.grid_plane`/`grid_offset`,
            # shared with the debug mask kernel -- see the comment there. This
            # kernel supplies only which plane a workgroup is on.
            _off_zh = fx.Int32(z_i32) * fx.Int32(num_head_q) + fx.Int32(head_q)
            _ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            _ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
            fmha.philox_report(
                philox_seed_output, philox_offset_output, _ph_seed, _ph_off
            )
            _ph_base, _ph_stride = PHILOX.grid_plane(
                _ph_off, _off_zh, max_seqlen_q, max_seqlen_k
            )

        # `clamp` is what bounds a KV row against `seqlen_k`. Nothing else does,
        # so the only way to drop it is the explicit unsafe knob.
        #
        # There used to be a second escape -- "with both prefetch distances 0
        # and neither load guard, there is no over-read" -- and both halves of
        # it were wrong:
        #
        #   * the only guard in the distance-0 `stage` path is
        #     `row < block_rows` with `block_rows == BLOCK_N`. That bounds the
        #     row *within the tile*; it says nothing about `seqlen_k`.
        #   * "the tail is masked" masks the *scores*, not the address. It held
        #     only while the interface padded `seqlen_k`, and that padding is
        #     gone.
        #
        # The ragged tail tile therefore walks rows `_full_end ..
        # _full_end + BLOCK_N - 1`, of which up to `BLOCK_N - 1` are past
        # `seqlen_k`; at the last KV head that leaves the allocation. Measured
        # 4096 bytes past K at B=1 H=512 seqlen 16 head_dim 128 with both
        # distances pinned to 0 -- a HIP fault at exactly K's end.
        #
        # It cannot be repaired by narrowing, because raggedness is a *runtime*
        # property and this predicate is `const_expr`. Note also that it tested
        # `V_NEEDS_GUARD` while the transposed-V path's guard is
        # `V_TR_NEEDS_GUARD`, and that for transposed V the global load sits
        # outside `publish_transposed`'s guard entirely -- two more ways for one
        # tensor's guard to answer for another.
        #
        # No policy-resolved build loses anything: `v_prefetch_dist` defaults to
        # 1 and no policy path sets it to 0, so the escape was unreachable
        # except by pinning the knob by hand. This emits identical code for
        # every shipped configuration.
        _KV_CLAMP = not _NO_KV_CLAMP
        _addr_kw = dict(
            seqlen_k=seqlen_k_v,
            seq_last=seq_last,
            hoist=KV_ADDR_HOIST,
            clamp=_KV_CLAMP,
        )
        q_tbase, q_toff, _ = fmha.make_addr_pair(
            q_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw
        )
        _, _, k_addr = fmha.make_addr_pair(
            k_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw
        )
        _, _, v_addr = fmha.make_addr_pair(
            v_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw
        )
        o_tbase, o_toff, _ = fmha.make_addr_pair(
            o_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw
        )

        # ---- PADDED_HEAD column handling ----
        #
        # Exactly one rule: an element is valid iff its column < hdim. It covers
        # both invalid regions without the kernel ever knowing the pitch:
        #
        #   [hdim, ceil8(hdim))   pad inside the allocation. Safe to load (the
        #                         chunk containing hdim ends at ceil8(hdim),
        #                         which is <= pitch by the contract), but the
        #                         contents are not guaranteed zero.
        #   [ceil8(hdim), tile)   past the row entirely. In BHSD these bytes
        #                         belong to head h+1, and at the last head of
        #                         the last token they are past the allocation.
        #                         Must not be addressed.
        #
        # So: a chunk whose *start* is >= hdim is redirected to column 0 (always
        # valid) and then masked away wholesale; a chunk that straddles is
        # loaded as-is and masked per element. Both fall out of the same two
        # operations, which is why there is no case analysis below.
        _hdim_qk_i = fx.Index(hdim_qk)
        _hdim_vo_i = fx.Index(hdim_vo)

        # The two column axes: QK accesses are bounded by hdim_qk, V/O by
        # hdim_vo. `PADDED_HEAD` false means the extent is a multiple of the
        # access width, so every access is wholly in range and the masking
        # compiles away.
        #
        qk_cols = fmha.MaskedAxis(_hdim_qk_i, active=PADDED_HEAD, elem_dtype=elem_dtype)
        vo_cols = fmha.MaskedAxis(_hdim_vo_i, active=PADDED_HEAD, elem_dtype=elem_dtype)

        # ---- The staged apertures ----
        #
        # K and V, the two tensors that transit LDS. Each carries the
        # cooperative-load geometry `_load_geom` computed for *its own* width,
        # so a site that gates on a guard has to name whose guard it is. `rows`
        # is absent because their row bound lives in `k_addr` / `v_addr` -- see
        # the `Aperture` docstring.
        #
        # Q and O get theirs when the per-wave load and the epilogue store move
        # behind the same interface; until then they would be objects with no
        # reader.
        k_ap = fmha.Aperture(
            qk_cols,
            lds_base=0,
            lds_stride=K_STRIDE,
            vec_width=VEC_WIDTH,
            threads_per_row=K_TPR_LOAD,
            rows_per_batch=K_ROWS_PER_BATCH,
            num_batches=NUM_BATCHES_K,
            needs_guard=K_NEEDS_GUARD,
        )
        v_ap = fmha.Aperture(
            vo_cols,
            lds_base=LDS_V_BASE,
            lds_stride=VT_STRIDE if V_TRANSPOSED else V_STRIDE,
            vec_width=VEC_WIDTH,
            threads_per_row=V_TPR_LOAD,
            rows_per_batch=V_ROWS_PER_BATCH,
            num_batches=NUM_BATCHES_V,
            needs_guard=V_NEEDS_GUARD,
        )

        def _load_global_half_vec(ptr, base64, off32, vec_type):
            return _pointer_load(
                vec_type, fmha.split_ptr(ptr, base64, off32, elem_type)
            )

        def _store_global_half(ptr, base64, off32, val):
            _pointer_store(val, fmha.split_ptr(ptr, base64, off32, elem_type))

        def load_global_f16xN(base_ptr, base64, off32):
            return _load_global_half_vec(base_ptr, base64, off32, vxf16_type)

        def load_global_v8f16(base_ptr, base64, off32):
            return _load_global_half_vec(base_ptr, base64, off32, v8f16_type)

        # How each tensor is read: its address split, paired with the load
        # instruction that consumes it. Each is `start -> (row, col) -> value`,
        # which is what every movement helper in `fmha_common` takes, so none
        # of them has to know that V^T uses a different instruction from V.
        fetch_k = fmha.reader(k_addr, lambda b, o: load_global_f16xN(k_ptr, b, o))
        fetch_v = fmha.reader(v_addr, lambda b, o: load_global_f16xN(v_ptr, b, o))
        fetch_v_tr = fmha.reader(
            v_addr, lambda b, o: fmha.global_load_tr_v8(v_ptr_i64, b, o, v8f16_type)
        )

        # ---- K staging ----

        def coop_load_k_global(start_k):
            """Issue this thread's K global loads; results stay in registers."""
            return fmha.read_batches(
                k_ap, fetch_k(start_k), load_row_in_batch, load_col_base
            )

        def coop_store_k_lds(vecs):
            """Distance-1 K staging: publish the tile loaded last iteration."""
            fmha.publish(
                k_ap,
                lds_kv,
                vecs,
                load_row_in_batch,
                load_col_base,
                fx.Index(BLOCK_N),
            )

        def coop_load_store_k(start_k):
            """Distance-0 K staging: load and store inside a single guard."""
            fmha.stage(
                k_ap,
                lds_kv,
                fetch_k(start_k),
                load_row_in_batch,
                load_col_base,
                fx.Index(BLOCK_N),
            )

        # ---- V staging ----
        #
        # Two layouts. Transposed stages V^T[d][kv] via global_load_tr_b128 so
        # GEMM2 reads one contiguous vector per operand; row-major stages
        # V[kv][d] and GEMM2 gathers 8 strided scalars. The two differ in the
        # helper they call and in nothing else here.
        #
        # The lane offsets are two distinct mappings. The *load* pair is the
        # address each lane must supply so the hardware transpose lands the
        # right 16(d) x 16(kv) block in WMMA-operand order (derivation in
        # `fmha.global_load_tr_v8`): within a group of 8 lanes the lane index
        # picks the kv row and the group index picks the 8-wide d half. The
        # *store* pair is where the lane's transposed result then belongs.
        v_tr = fmha.TransposedTiling(
            d_blocks=V_TR_D_BLOCKS,
            tiles=_V_TR_TILES,
            loads=V_TR_LOADS,
            needs_guard=V_TR_NEEDS_GUARD,
            num_waves=NUM_WAVES,
            d_step=WMMA_N,
            kv_step=WMMA_K,
            wave_id=wave_id,
            load_d_off=((lane // 8) % 2) * WMMA_LANE_K,
            load_kv_off=(lane // 16) * WMMA_LANE_K + (lane % 8),
            store_d_off=lane16,
            store_kv_off=klane * WMMA_LANE_K,
        )

        def coop_load_v_global(start_k, chunk=0):
            """V columns [chunk*VO_CHUNK_COLS, +VO_CHUNK_COLS) of this KV tile."""
            if const_expr(V_TRANSPOSED):
                return fmha.read_transposed(
                    v_ap,
                    v_tr,
                    fetch_v_tr(start_k),
                    chunk * VO_CHUNK_COLS + D_OFFSET,
                )
            col = v_col_base
            if const_expr(D_OFFSET):
                col = fx.Index(D_OFFSET) + col
            return fmha.read_batches_unmasked(
                v_ap, fetch_v(start_k), v_row_in_batch, col
            )

        def coop_store_v_lds(vecs):
            if const_expr(V_TRANSPOSED):
                fmha.publish_transposed(v_ap, v_tr, lds_kv, vecs)
            else:
                fmha.publish(
                    v_ap,
                    lds_kv,
                    vecs,
                    v_row_in_batch,
                    v_col_base,
                    fx.Index(BLOCK_N),
                )

        # ---- Q preload ----
        # One row per row sub-tile; at ROW_SUBTILES == 1 this is the
        # single-tile mapping.
        q_rows = [
            start_q + wave_q_offset + fx.Index(qt * WMMA_M) + lane16
            for qt in range_constexpr(ROW_SUBTILES)
        ]
        q_row_i32s = [fx.Int32(r) for r in q_rows]
        # Intra-tile Q rows, bounded by BLOCK_M so the 32-bit offset stays small.
        q_rows_in_tile = [
            wave_q_offset + fx.Index(qt * WMMA_M) + lane16
            for qt in range_constexpr(ROW_SUBTILES)
        ]
        # Rows are bounded by the real Q length; a workgroup's last tile can
        # hang past it. Unlike the column axes this is never inactive. Q is
        # read straight into registers, so its aperture has no LDS placement
        # and no cooperative geometry.
        q_ap = fmha.Aperture(qk_cols, rows=fmha.MaskedAxis(fx.Index(seqlen_q_i32)))
        q_tile_base = q_tbase(_q_start_addr)

        def fetch_q(row, col):
            return load_global_v8f16(q_ptr, q_tile_base, q_toff(row, col))

        q_in_bounds_all = []
        q_b_packs_all = []
        for qt in range_constexpr(ROW_SUBTILES):
            # The index-typed row, deliberately, not `q_row_i32s[qt]`. The i32
            # copy exists for the causal mask; gating against it here would
            # start its live range early, and at the widest causal builds that
            # overlap spills -- BLOCK_DMODEL 384 causal paid 16 more bytes of
            # scratch and 6% throughput. `MaskedAxis.gate` emits the signed
            # compare that `fx.Index` being unsigned would otherwise deny.
            #
            # Once per row, not once per column: that is the same live-range
            # argument, and it is why `read_v8` takes the gate rather than
            # recomputing it.
            _in, _safe = q_ap.rows.gate(q_rows[qt], q_rows_in_tile[qt])
            _packs = []
            for ks in range_constexpr(K_STEPS_QK):
                q_col = shard_qk_off + fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
                _packs.append(q_ap.read_v8(fetch_q, _safe, q_col, _in))
            q_in_bounds_all.append(_in)
            q_b_packs_all.append(_packs)

        # ---- Constants ----
        # Mask fill: genuinely -inf, so exp2(-inf - m) is exactly 0.
        c_neg_inf = fx.Float32(float("-inf"))
        # m_i floor: finite, so an all-masked tile cannot produce -inf - -inf.
        # Finite floor, so an all-masked row cannot produce -inf - -inf.
        c_m_init = fx.Float32(-3.40282e38)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = sm_log2e
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)

        def reduction_peer(v_f32):
            return gpu.shuffle_xor(fx.Float32(v_f32), shuf_16_i32, width_i32)

        # Right edge of the visible band: key j is visible to query i only if
        # j <= i + _diag. This *is* `window_right` -- 0 for top-left,
        # seqlen_k - seqlen_q for bottom-right, and an explicit argument under
        # gSWA. It is signed: with seqlen_q > seqlen_k it goes negative, and
        # whole leading Q rows then see no keys at all.
        #
        # Everything derived from a window stays fx.Int32 with explicit signed
        # predicates, per sdpa-gswa-plan.md section 2.4. fx.Index is unsigned,
        # so a negative window reaching it silently becomes enormous.
        if const_expr(CAUSAL):
            # Sentinels resolve per sequence, not on the host -- see
            # `fmha.resolve_window`. Everything derived from a window stays
            # i32 from here down: bounds go negative, and `fx.Index` is
            # unsigned.
            _wl_i32, _wr_i32 = fmha.resolve_window(
                window_left, window_right, seqlen_q_i32, seqlen_k_i32
            )

        # ---- Split the KV range into full and masked regions ----
        #
        # Emitting the regions as separate loops means the masks exist only in
        # the masked one -- `MASK_STEPS` in AOTriton's terms, with the split
        # structural rather than a per-tile branch.
        #
        # **How many regions there are depends on the arm.** Non-causal has
        # two, `[full][tail-masked]`, because the only thing that can cut a
        # tile is running past seqlen_k. Causal has *three*: a left window
        # kills columns at the start of the range as well, so masked tiles are
        # a prefix as well as a suffix and tile 0 is not automatically live.
        # A negative `window_left` is the sharpest case -- it pushes the whole
        # band right of the diagonal, so the leading masked run can span
        # several tiles rather than clipping one.
        #
        # Do not carry the two-region intuition into the causal branch; the
        # "every earlier tile is fully live" shortcut is true only above.
        #
        # This is what pays back the unconditional mask P1e had to use: a
        # dynamic `scf.if` guard inside one loop measured much worse than no
        # guard at all, because the region boundary blocks scheduling in a
        # latency-bound loop. A static split has no such boundary.
        if const_expr(CAUSAL):
            # ---- gSWA: three regions over one contiguous block range ----
            #
            # A left window can kill columns in any tile, including tile 0, so
            # the masked tiles are a prefix as well as a suffix:
            #
            #     [ left-masked ][ full ][ right-masked ]
            #
            # The three are *contiguous and non-overlapping by construction*
            # here, because they are derived by cutting one visited range
            # rather than intersected as three independent intervals. That
            # collapses two of the three special cases in sdpa-gswa-plan.md
            # section 2.2: a window narrower than a block leaves the full
            # region empty, which is detected once and turns the other two
            # into a single masked run, and an irregular seqlen_q needs no
            # special handling because `_q_hi` already bounds the rows.
            #
            # Bounds for the rows this Q block actually owns, [start_q, q_hi):
            #   a column c is live for row i iff  i - w_left <= c <= i + w_right
            # so over the whole block the live columns span
            #   [ start_q - w_left, (q_hi - 1) + w_right ]
            # and a tile is *fully* live iff every one of its columns is live
            # for every row -- worst case the largest row on the left and the
            # smallest on the right.
            _regions = fmha.decompose_causal_regions(
                start_q,
                seqlen_q_i32,
                seqlen_k_i32,
                _wl_i32,
                _wr_i32,
                BLOCK_M,
                BLOCK_N,
                _alive,
            )
            _BN_I32 = fx.Int32(BLOCK_N)
            _n_l, _n_f, _n_r = _regions.n_left, _regions.n_full, _regions.n_right
            _l_col0, _f_col0 = _regions.left_col0, _regions.full_col0
            _r_col0, _m_col0 = _regions.right_col0, _regions.masked_col0
            _n_masked = fx.Index(_n_l + _n_r)
        else:
            # No mask beyond the KV tail: one full region, then one partial
            # tile. `kv_upper` is rounded *up* to a whole BLOCK_N because the
            # loop steps by BLOCK_N, and a bound that is not a multiple of it
            # drops the final partial tile entirely -- at seqlen 40 with
            # BLOCK_N 32 the kernel attended to only the first 32 keys, which
            # is wrong for every row rather than just the tail.
            _full_end = (seqlen_k_v // fx.Index(BLOCK_N)) * fx.Index(BLOCK_N)
            kv_upper = fx.Index(
                ((seqlen_k_v + fx.Index(BLOCK_N - 1)) // fx.Index(BLOCK_N))
                * fx.Index(BLOCK_N)
            )
            # Same empty-work clamp as the causal arm above.
            _full_end = fx.Index(_full_end if _alive else fx.Index(0))
            kv_upper = fx.Index(kv_upper if _alive else fx.Index(0))

        # Tiles this workgroup will actually walk. Zero for a sequence with no
        # keys and for every workgroup the varlen grid dispatches past the end
        # of a short one.
        if const_expr(CAUSAL):
            _kv_tiles_i32 = _n_f + _n_l + _n_r
        else:
            _kv_tiles_i32 = fx.Int32(kv_upper)

        # ---- Prologue: at distance 1, the first tile's K / V go to registers ----
        #
        # "First" is tile 0 for everything except gSWA, where a left window
        # can make the first *visited* tile any block -- loading tile 0 there
        # feeds the first iteration the wrong K and V. That was worth a
        # relative error of 1.3 on a shape whose interval arithmetic was
        # already correct, which is the failure mode plan section 5 predicted:
        # a prefetch bug survives every check that only looks at the loop
        # bounds.
        if const_expr(CAUSAL):
            _first_col = fx.Index(
                common_utils.smax(
                    _f_col0 if _n_f > fx.Int32(0) else _m_col0,
                    fx.Int32(0),
                )
            )
        else:
            _first_col = fx.Index(0)
        # **Issued only if a KV tile will actually be walked.**
        #
        # The prefetch runs before any loop bound is consulted, so a workgroup
        # with nothing to do -- a sequence with no keys, or one of the dead
        # workgroups varlen's Max_seqlen_q-sized grid dispatches past the end
        # of a short sequence -- would otherwise still issue a K and a V tile
        # load and throw the result away.
        #
        # Clamping the address instead would make that load land somewhere
        # harmless, which fixes the symptom: the load should not happen. And
        # at seqlen_k == 0 there is no harmless address to clamp *to* -- row 0
        # of an empty sequence is one past its end.
        #
        # A 0-or-1-trip `range(..., init=...)` is how this kernel already
        # expresses predicated state: FlyDSL's dynamic `if` merges named
        # scalars only and rejects the list of vectors a prefetch produces,
        # while a loop carries exactly that list. The trip count is uniform
        # across the workgroup, so no divergence is introduced.
        _pf_n = fx.Index(fx.Int32(1) if _kv_tiles_i32 > fx.Int32(0) else fx.Int32(0))
        _pf_init = []
        if const_expr(K_PREFETCH_DIST):
            for _ in range_constexpr(k_ap.num_batches):
                _pf_init.append(Vec.filled(VEC_WIDTH, 0.0, elem_dtype).ir_value())
        if const_expr(V_PREFETCH_DIST):
            for _ in range_constexpr(V_LOADS):
                _pf_init.append(Vec.filled(VEC_WIDTH, 0.0, elem_dtype).ir_value())
        _pf = _pf_init
        if const_expr(K_PREFETCH_DIST or V_PREFETCH_DIST):
            for _pfi, _pf_args in range(fx.Index(0), _pf_n, 1, init=_pf_init):
                _y = []
                if const_expr(K_PREFETCH_DIST):
                    _y.extend(coop_load_k_global(_first_col))
                if const_expr(V_PREFETCH_DIST):
                    _y.extend(coop_load_v_global(_first_col))
                _pf = yield _y
            # `scf_yield_` returns a bare value, not a list, when the loop
            # carries exactly one -- which happens at BLOCK_DMODEL 16, where the K
            # prefetch is off and V is a single load. Indexing that bare value
            # extracts a vector *element*, so the next use sees an f16 scalar
            # where it wants a vector, and the failure surfaces far away in
            # the LDS store.
            if const_expr(len(_pf_init) == 1):
                _pf = [_pf]
        if const_expr(K_PREFETCH_DIST):
            _k_vecs_init = [_pf[_i] for _i in range_constexpr(k_ap.num_batches)]
        if const_expr(V_PREFETCH_DIST):
            _off = k_ap.num_batches if K_PREFETCH_DIST else 0
            _v_vecs_init = [_pf[_off + _i] for _i in range_constexpr(V_LOADS)]

        # Loop-carried state layout:
        #   [0 .. 2*ROW_SUBTILES)             m/l per Q row sub-tile, interleaved
        #   [_ML .. _ML + ROW_SUBTILES*O_ACCS) O accumulators per Q row sub-tile
        #   [_OFF ..)                         K vectors (distance 1 only), then V
        _ML = 2 * ROW_SUBTILES
        _OFF = _ML + ROW_SUBTILES * O_ACCS
        _KOFF = _OFF
        _VOFF = _OFF + (k_ap.num_batches if K_PREFETCH_DIST else 0)

        init_args = []
        for _ in range_constexpr(ROW_SUBTILES):
            init_args.append(fx.as_ir_value(c_m_init))
            init_args.append(fx.as_ir_value(c_zero_f))
        for _ in range_constexpr(ROW_SUBTILES * O_ACCS):
            init_args.append(fx.as_ir_value(c_zero_v8f32))
        if const_expr(K_PREFETCH_DIST):
            for batch in range_constexpr(k_ap.num_batches):
                init_args.append(_k_vecs_init[batch])
        if const_expr(V_PREFETCH_DIST):
            for batch in range_constexpr(V_LOADS):
                init_args.append(_v_vecs_init[batch])

        loop_results = init_args

        def kv_loop_body(
            kv_block_start, inner_iter_args, _MASK_STEPS, next_kv_start=None
        ):
            """One KV tile. `_MASK_STEPS` is a Python bool resolved at trace
            time, so the masked and unmasked regions emit different code.

            `next_kv_start` is the tile the distance-1 prefetch should fetch.
            It defaults to the following tile, which is right whenever the
            region being walked is contiguous. gSWA's masked loop walks two
            disjoint runs, so it passes the piecewise successor explicitly --
            getting that wrong fetches the wrong tile and is invisible to a
            correctness test whenever the value is overwritten before use."""
            m_run = [inner_iter_args[2 * qt] for qt in range_constexpr(ROW_SUBTILES)]
            l_run = [
                inner_iter_args[2 * qt + 1] for qt in range_constexpr(ROW_SUBTILES)
            ]
            o_accs_all = [
                [
                    inner_iter_args[_ML + qt * O_ACCS + i]
                    for i in range_constexpr(O_ACCS)
                ]
                for qt in range_constexpr(ROW_SUBTILES)
            ]
            if const_expr(K_PREFETCH_DIST):
                _k_vecs_cur = [
                    inner_iter_args[_KOFF + b]
                    for b in range_constexpr(k_ap.num_batches)
                ]
            if const_expr(V_PREFETCH_DIST):
                _v_vecs_cur = [
                    inner_iter_args[_VOFF + b] for b in range_constexpr(V_LOADS)
                ]

            if const_expr(next_kv_start is None):
                next_kv_start = kv_block_start + fx.Index(BLOCK_N_OUT)

            # At distance 1 this tile's K is already in registers: publish it,
            # then immediately issue the *next* tile's K load so it is in flight
            # across GEMM1 + softmax rather than being waited on right here.
            # At distance 0 the load and store are adjacent and the latency is
            # exposed -- that is the whole difference between the two schedules.
            if const_expr(K_PREFETCH_DIST):
                coop_store_k_lds(_k_vecs_cur)
                gpu.barrier()
                _k_vecs_next = coop_load_k_global(next_kv_start)
            else:
                coop_load_store_k(kv_block_start)
                gpu.barrier()

            # ==== GEMM1: S = K @ Q^T ====
            # At ROW_SUBTILES > 1 each K pack feeds every row sub-tile's S
            # accumulators: one LDS read serves ROW_SUBTILES WMMAs. That reuse is
            # the point of the knob.
            s_accs_all = [
                [fx.as_ir_value(c_zero_v8f32) for _ in range(NUM_S_ACCS)]
                for _ in range_constexpr(ROW_SUBTILES)
            ]

            for ks in range_constexpr(K_STEPS_QK):
                k_col = shard_qk_off + fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K

                for st_idx in range_constexpr(COL_SUBTILES):
                    st_base_row = st_idx * COLS_PER_SUBTILE

                    k_row_a = lane16 + fx.Index(st_base_row)
                    k_pack_a = k_ap.from_lds(lds_kv, k_row_a, k_col)

                    k_row_b = lane16 + fx.Index(st_base_row + 16)
                    k_pack_b = k_ap.from_lds(lds_kv, k_row_b, k_col)

                    acc_idx_a = st_idx * 2
                    acc_idx_b = st_idx * 2 + 1
                    for qt in range_constexpr(ROW_SUBTILES):
                        s_accs_all[qt][acc_idx_a] = fmha.wmma_acc(
                            k_pack_a, q_b_packs_all[qt][ks], s_accs_all[qt][acc_idx_a]
                        )
                        s_accs_all[qt][acc_idx_b] = fmha.wmma_acc(
                            k_pack_b, q_b_packs_all[qt][ks], s_accs_all[qt][acc_idx_b]
                        )

            # ==== Cross-shard S reduction ====
            # Each shard-wave holds a partial sum over its own BLOCK_DMODEL slice;
            # the full S is their sum. Explicit partials, not ds_add_f32:
            # measured 54 vs 1055 WMMA-equivalents, see
            # kernels/microbench/lds_reduce.py.
            if const_expr(QK_SHARDS > 1):
                for qt in range_constexpr(ROW_SUBTILES):
                    s_accs_all[qt] = fmha.reduce_s_across_shards(
                        s_accs_all[qt],
                        lds_byte_base=_lds_byte_base,
                        byte0=_RED_BYTE0,
                        wave_id=wave_id,
                        lane=lane,
                        shard_id=shard_id,
                        q_tile_in_block=q_tile_in_block,
                        num_shards=QK_SHARDS,
                        f32_per_wave=RED_F32_PER_WAVE,
                        warp_size=WARP_SIZE,
                        fastmath=fastmath,
                    )

            # ==== Online softmax, per Q row sub-tile ====
            # Each row sub-tile keeps its own running max/sum and its own O
            # accumulators.
            m_new_all, l_new_all, p_vals_all = [], [], []
            for qt in range_constexpr(ROW_SUBTILES):
                s_accs = s_accs_all[qt]
                q_row_i32 = q_row_i32s[qt]
                m_running = m_run[qt]
                l_running = l_run[qt]

                # ---- KV tail mask: columns >= seqlen_k are not real keys ----
                #
                # seqlen is never padded, so the final KV tile of a ragged
                # sequence covers columns past the end. `kv_addr` clamps those
                # *addresses* so the loads stay in bounds, which means they read
                # a duplicate real row -- safe, but the scores are garbage and
                # would enter the softmax. Mask them to -inf here.
                #
                # Done on the eight-wide accumulators rather than the unpacked
                # scalars: NUM_S_ACCS is at most 8 (BLOCK_N 128), so the
                # branch's live set stays small, and one vector select replaces
                # eight scalar ones.
                #
                # Guarded, so only the tail tile pays. Interior tiles cost a
                # single scalar compare. (This is `MASK_STEPS` with a dynamic
                # guard instead of a structural one; P2's interval
                # decomposition replaces it with the structural form.)
                s_raw = []
                for st in range_constexpr(NUM_S_ACCS):
                    for r in range_constexpr(8):
                        s_raw.append(Vec(s_accs[st])[r])

                # Scale before the row max, so m_i lives in the scaled
                # domain and the exponent is a plain subtract.
                s_raw = [fastmath.mul(v, c_sm_scale_log2e) for v in s_raw]

                if const_expr(BIAS_TYPE):
                    # ---- Bias, after the scale and before the mask ----
                    #
                    # After the scale because m_i and the exponent live in the
                    # base-2 scaled domain, so a bias in natural units has to
                    # be multiplied by log2(e) first -- which is what
                    # AOTriton's `qk += bias * 1.44269504089` is doing.
                    #
                    # Before the mask so a column past seqlen_k stays -inf
                    # rather than becoming -inf + bias. Those columns are not
                    # keys the caller hid; they do not exist, and neither do
                    # their bias entries.
                    #
                    # Not a gather. Element i of the flattened accumulators is
                    # KV column (i//16)*32 + ((i//8)%2)*16 + klane*8 + i%8, so
                    # within each group of eight only i%8 varies: those eight
                    # are eight *contiguous* columns starting at klane*8, and
                    # one v8 load covers them -- the same shape as the K and V
                    # loads.
                    # **The row is clamped, and this is the one that faulted.**
                    # A workgroup covers BLOCK_M query rows whatever `seqlen_q`
                    # is -- 128 of them at head_dim 24 against a seqlen_q of 11
                    # -- and every one of those dead lanes still runs this load.
                    # Unclamped, row 33 of an 11-row plane is 22 rows past it,
                    # and `stride_b_seq_q` rows are far enough apart that it
                    # leaves the tensor entirely:
                    #
                    #   fault 0x7ef51f810000, 70,074 B past a 519,750 B
                    #   allocation = 3*5*11*1575*2, the bias exactly
                    #
                    # Not something a poisoned margin can catch either: a row
                    # index one too large lands on the next row's *real data*,
                    # so it reads plausible numbers until it runs off the end.
                    #
                    # `q_rows[qt]` and not `gate`'s `_safe`: `MaskedAxis.safe`
                    # returns the offset *within the tile* for an addressed
                    # access, and the bias is indexed absolutely from
                    # `_b_base`. The dQ kernel shipped that exact confusion
                    # once. `_b_base` already carries `_q_row_off_v`, so this
                    # stays correct under varlen.
                    _bq = q_rows[qt] if (q_row_i32s[qt] < seqlen_q_i32) else fx.Index(0)
                    _b_row = _b_base + _bq * fx.Index(stride_b_seq_q)
                    # The KV column offset, the mirror of `_bq` on the row.
                    _b_col0 = _k_row_off_v
                    # Modification 5: an f32 mask is read as f32, and a key axis
                    # whose stride is not 1 takes this per-element path on every
                    # tile, with the column scaled by its stride.
                    _b_elem = ir.F32Type.get() if BIAS_F32 else elem_type
                    if const_expr(_MASK_STEPS or BIAS_KSTRIDED):
                        # **The tail tile reads one element at a time, with the
                        # address clamped.** A group's eight columns can run
                        # past `seqlen_k` here, and the v8 below would issue the
                        # whole 16-byte access anyway: the mask further down
                        # replaces those *scores* with -inf, so the wrong value
                        # was harmless and the out-of-bounds *access* was
                        # invisible -- until it reached a page the tensor does
                        # not own. That is a real fault, not a theoretical one
                        # (AOTriton `test_irregulars` seqlen_k 1063, hdim 24:
                        # "Memory access fault ... kernel:
                        # flash_attn_func_aiw_kernel_0").
                        #
                        # Sliding the eight-wide base back into the row instead
                        # would be wrong -- it shifts which column each element
                        # carries, so the live ones would take a neighbour's
                        # bias. Clamping *per element* is exact, because a dead
                        # element's value is discarded either way. Same
                        # contract dK/dV's bias load already has.
                        #
                        # Cost lands only here: full tiles are bounded by
                        # `blk_last_whole`, so every column of one exists, and
                        # `_MASK_STEPS` is `const_expr` so they keep the v8.
                        _kv_i32 = fx.Int32(kv_block_start)
                        _klane_off = fx.Int32(klane) * fx.Int32(8)
                        for _i in range_constexpr(NUM_S_VALS):
                            _bcol = (
                                _kv_i32
                                + fx.Int32(fmha.acc_elem_column(_i))
                                + _klane_off
                            )
                            _bin = _bcol < seqlen_k_i32
                            _bsafe = _b_col0 + fx.Index(
                                fx.Int32(_bcol) if _bin else fx.Int32(0)
                            )
                            if const_expr(BIAS_KSTRIDED):
                                _bsafe = _bsafe * fx.Index(stride_b_seq_k)
                            _braw = fx.Float32(
                                fx.as_dsl_value(
                                    _pointer_load(
                                        _b_elem,
                                        buffer_ops.get_element_ptr(
                                            _b_ptr,
                                            fx.Int64(_b_row + _bsafe),
                                            elem_type=_b_elem,
                                        ),
                                    )
                                ).to(fx.Float32)
                            )
                            _bs = fx.Float32(_braw if _bin else fx.Float32(0.0))
                            s_raw[_i] = fastmath.add(
                                s_raw[_i], fastmath.mul(_bs, fx.Float32(_LOG2E))
                            )
                    else:
                        for _st in range_constexpr(NUM_S_ACCS):
                            _c0 = (_st // 2) * 32 + (_st % 2) * 16
                            _b_off = (
                                _b_row
                                + _b_col0
                                + fx.Index(kv_block_start)
                                + fx.Index(_c0)
                                + klane * fx.Index(8)
                            )
                            if const_expr(BIAS_F32):
                                _bv = _pointer_load(
                                    Vec.make_type(8, fx.Float32),
                                    fmha.split_ptr(
                                        _b_ptr, _b_off, fx.Index(0), _b_elem
                                    ),
                                )
                            else:
                                _bv = load_global_v8f16(_b_ptr, _b_off, fx.Index(0))
                            for _r in range_constexpr(8):
                                _bs = fx.Float32(Vec(_bv)[_r].to(fx.Float32))
                                s_raw[_st * 8 + _r] = fastmath.add(
                                    s_raw[_st * 8 + _r],
                                    fastmath.mul(_bs, fx.Float32(_LOG2E)),
                                )

                if const_expr(_MASK_STEPS):
                    # ---- Masked region: KV tail and causal, fused ----
                    #
                    # Only tiles in the masked region reach this. Full tiles
                    # are emitted by the other region with no mask at all,
                    # which is the entire point of the split: masking used to
                    # be paid on every tile.
                    #
                    # Element i of the flattened accumulators is KV column
                    #   (i//16)*32 + ((i//8)%2)*16 + klane*8 + i%8
                    # -- the GEMM1 unroll walks (sub-tile, half) pairs, and
                    # within a 16-row WMMA block a lane holds rows
                    # klane*8 + si.
                    #
                    # Two conditions, one select: a column is dead if it is
                    # past seqlen_k, or (causal) beyond this row's diagonal.
                    #
                    # **Must run after the sm_scale multiply.** The kernel is
                    # built with nnan but *not* ninf precisely so an -inf can
                    # survive arithmetic; even so, keeping the mask after the
                    # scale avoids relying on that. Masking before it silently
                    # did nothing when ninf was still enabled.
                    #
                    # No `scf.if` guard: a runtime "does this tile need it"
                    # branch measured far worse than just doing the selects
                    # (BLOCK_DMODEL 192 non-causal 78.0 against 98.3 TFLOPS), the
                    # region boundary blocking scheduling in a latency-bound
                    # loop. The region split gives the same benefit statically.
                    _kv_i32 = fx.Int32(kv_block_start)
                    _klane_off = fx.Int32(klane) * fx.Int32(8)
                    _seq_i32 = seqlen_k_i32
                    for _i in range_constexpr(NUM_S_VALS):
                        _col = _kv_i32 + fx.Int32(fmha.acc_elem_column(_i)) + _klane_off
                        _dead = _col >= _seq_i32
                        if const_expr(CAUSAL):
                            # Both edges of the band. Signed throughout:
                            # q_row - w_left is negative for every row when
                            # w_left is "unbounded" (== seqlen_q), which is how
                            # plain causal maps onto this path with the left
                            # term inert.
                            _dead = _dead | (_col > q_row_i32 + _wr_i32)
                            _dead = _dead | (_col < q_row_i32 - _wl_i32)
                        s_raw[_i] = c_neg_inf if _dead else s_raw[_i]

                local_max = s_raw[0]
                for r in range_constexpr(NUM_S_VALS - 1):
                    local_max = fastmath.max(local_max, s_raw[r + 1])
                peer_max = reduction_peer(local_max)
                row_max = fastmath.max(local_max, peer_max)
                m_new_raw = fastmath.max(m_running, row_max)

                # m is already scaled, so no scale appears in either exponent.
                corr = rocdl.exp2(
                    ir.F32Type.get(), fx.as_ir_value(fastmath.sub(m_running, m_new_raw))
                )
                neg_m = fastmath.sub(c_zero_f, m_new_raw)

                p_vals = []
                local_sum = fx.as_ir_value(c_zero_f)
                for r in range_constexpr(NUM_S_VALS):
                    diff = fastmath.add(s_raw[r], neg_m)
                    p = rocdl.exp2(ir.F32Type.get(), fx.as_ir_value(diff))
                    p_vals.append(p)
                    local_sum = fastmath.add(local_sum, p)

                peer_sum = reduction_peer(local_sum)
                tile_sum = fastmath.add(local_sum, peer_sum)
                l_corr = fastmath.mul(corr, l_running)
                l_new = fastmath.add(l_corr, tile_sum)

                corr_vec = (
                    Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
                )
                for dc in range_constexpr(O_ACCS):
                    o_accs_all[qt][dc] = fastmath.mul(o_accs_all[qt][dc], corr_vec)

                if const_expr(ENABLE_DROPOUT):
                    # **After `l_new`, before the O accumulation.** The softmax
                    # denominator must be the *undropped* sum, or the result
                    # stops being an expectation of the undropped attention and
                    # the logsumexp the backward pass reads is wrong. Reversing
                    # these two lines produces plausible output that is wrong by
                    # a per-row factor, and no shape check notices
                    # (sdpa-dropout-plan.md §6).
                    #
                    # A group of eight consecutive elements is eight contiguous
                    # KV columns (§2 of the bias plan has the same identity), so
                    # each group is one span of the stream.
                    for _st in range_constexpr(NUM_S_ACCS):
                        _c0 = (_st // 2) * 32 + (_st % 2) * 16
                        _bcol = (
                            fx.Int64(kv_block_start)
                            + _c0
                            + fx.Int64(fx.Int32(klane) * 8)
                        )
                        _first = PHILOX.grid_offset(
                            _ph_base, _ph_stride, q_rows[qt], _bcol
                        )
                        _keep = PHILOX.keep_span(_ph_seed, _first, 8, idropout_p)
                        for _r in range_constexpr(8):
                            _i = _st * 8 + _r
                            p_vals[_i] = fx.as_ir_value(
                                fx.Float32(p_vals[_i]) if _keep[_r] else fx.Float32(0.0)
                            )

                m_new_all.append(m_new_raw)
                l_new_all.append(l_new)
                p_vals_all.append(p_vals)

            # ==== Build P packs, per Q row sub-tile ====
            p_packs_all_qt = []
            for qt in range_constexpr(ROW_SUBTILES):
                p_vals = p_vals_all[qt]
                p_packs_all = []
                for st_idx in range_constexpr(COL_SUBTILES):
                    p_packs_st = []
                    for pks in range_constexpr(PV_K_STEPS):
                        acc_idx = st_idx * 2 + pks
                        p_base = acc_idx * 8
                        p_slice = [p_vals[p_base + j] for j in range(8)]

                        if const_expr(dtype_str == "bf16"):
                            p_packs_st.append(
                                fmha.bf16_trunc_pack_v8(p_slice, elem_dtype)
                            )
                        else:
                            elem_list = []
                            for j in range_constexpr(8):
                                elem_list.append(fx.Float32(p_slice[j]).to(elem_dtype))
                            p_packs_st.append(
                                Vec.from_elements(elem_list, elem_dtype).ir_value()
                            )
                    p_packs_all.append(p_packs_st)
                p_packs_all_qt.append(p_packs_all)

            def _gemm2_chunk(_vc):
                """GEMM2 over the V window currently resident in LDS."""

                def _load_v(st_kv_base_val, pks_val, dc_val):
                    if const_expr(V_TRANSPOSED):
                        # V^T[d][kv]: the 8 kv values this lane needs are
                        # contiguous, so this is one vector read instead of 8
                        # strided scalar loads.
                        d_pos = shard_vo_off + fx.Index(dc_val * D_CHUNK) + lane16
                        kv0 = (
                            fx.Index(st_kv_base_val + pks_val * PV_K_STEP)
                            + klane * WMMA_LANE_K
                        )
                        return v_ap.from_lds(lds_kv, d_pos, kv0)
                    d_pos = fx.Index(dc_val * D_CHUNK) + lane16
                    v_elems = []
                    for k_sub in range_constexpr(8):
                        kv_row = (
                            fx.Index(st_kv_base_val + pks_val * PV_K_STEP)
                            + klane * WMMA_LANE_K
                            + fx.Index(k_sub)
                        )
                        v_elems.append(
                            fx.ptr_load(
                                lds_kv + fx.Int32(v_ap.lds_index(kv_row, d_pos))
                            )
                        )
                    return Vec.from_elements(v_elems, elem_dtype).ir_value()

                # Software pipeline: preload the first V pack, then prefetch the
                # next one while the current WMMA runs.
                cur_v_packs = []
                for st_idx in range_constexpr(COL_SUBTILES):
                    cur_v_packs.append(_load_v(st_idx * COLS_PER_SUBTILE, 0, 0))

                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        next_dc = dc + 1
                        next_pks = pks
                        if const_expr(next_dc >= D_CHUNKS):
                            next_dc = 0
                            next_pks = pks + 1
                        has_next = const_expr(next_pks < PV_K_STEPS)

                        next_v_packs = []
                        if const_expr(has_next):
                            for st_idx in range_constexpr(COL_SUBTILES):
                                next_v_packs.append(
                                    _load_v(
                                        st_idx * COLS_PER_SUBTILE, next_pks, next_dc
                                    )
                                )

                        # One V operand, ROW_SUBTILES WMMAs: this is what halves
                        # the V LDS reads per FLOP at ROW_SUBTILES > 1.
                        for st_idx in range_constexpr(COL_SUBTILES):
                            for qt in range_constexpr(ROW_SUBTILES):
                                o_accs_all[qt][_vc * D_CHUNKS + dc] = fmha.wmma_acc(
                                    cur_v_packs[st_idx],
                                    p_packs_all_qt[qt][st_idx][pks],
                                    o_accs_all[qt][_vc * D_CHUNKS + dc],
                                )

                        if const_expr(has_next):
                            cur_v_packs = next_v_packs

            if const_expr(VO_CHUNKS == 1):
                # Unchunked: keep the original shape exactly. Wrapping this in a
                # 1-trip chunk loop cost 4 VGPRs (190 -> 194), which crosses an
                # allocation-granularity boundary and drops occupancy 8 -> 7
                # waves/SIMD, measured at -4% on BLOCK_DMODEL 128.
                if const_expr(V_PREFETCH_DIST):
                    coop_store_v_lds(_v_vecs_cur)
                    gpu.barrier()
                    # Issue the next tile's V here, not after GEMM2: this way
                    # the load flies over GEMM2 instead of only over the loop
                    # back-edge. (The baseline kernel issues it after GEMM2;
                    # bp's placement is strictly better and is what aiw uses
                    # for both schedules.)
                    _v_vecs_next = coop_load_v_global(next_kv_start, 0)
                else:
                    coop_store_v_lds(coop_load_v_global(kv_block_start, 0))
                    gpu.barrier()
                _gemm2_chunk(0)
            else:
                _v_hold = _v_vecs_cur
                for vchunk in range_constexpr(VO_CHUNKS):
                    coop_store_v_lds(_v_hold)
                    gpu.barrier()
                    if const_expr(vchunk + 1 < VO_CHUNKS):
                        _v_hold = coop_load_v_global(kv_block_start, vchunk + 1)
                    else:
                        _v_vecs_next = coop_load_v_global(next_kv_start, 0)
                    _gemm2_chunk(vchunk)
                    if const_expr(vchunk + 1 < VO_CHUNKS):
                        # All waves must finish reading this window before the
                        # next chunk overwrites it.
                        gpu.barrier()

            _yield_args = []
            for qt in range_constexpr(ROW_SUBTILES):
                _yield_args.append(m_new_all[qt])
                _yield_args.append(l_new_all[qt])
            for qt in range_constexpr(ROW_SUBTILES):
                for i in range_constexpr(O_ACCS):
                    _yield_args.append(o_accs_all[qt][i])
            if const_expr(K_PREFETCH_DIST):
                for batch in range_constexpr(k_ap.num_batches):
                    _yield_args.append(_k_vecs_next[batch])
            if const_expr(V_PREFETCH_DIST):
                for batch in range_constexpr(V_LOADS):
                    _yield_args.append(_v_vecs_next[batch])
            return _yield_args

        loop_results = init_args
        if const_expr(CAUSAL):
            # Still **two** emitted bodies, not three. The body already costs
            # 63 VGPRs at BLOCK_DMODEL 128 and spills BLOCK_DMODEL 192 at two copies
            # (plan1 sections 6.2, 2.6), so the two masked runs share one loop
            # walked over a piecewise index -- one select per masked iteration,
            # paid only in the masked region.
            #
            # Order is full, then left-masked, then right-masked. Running the
            # full region *first* is what made a causal-equivalent window
            # bit-identical to the dedicated causal path that used to live
            # here: with an unbounded left window the left run is empty and
            # the order collapses to full-then-tail, exactly the pre-gSWA
            # split. Online softmax is order-independent mathematically but
            # not in floating point, so walking the masked runs first would
            # have been just as correct and would have lost that property --
            # which is the property that licensed deleting CAUSAL_TYPE 1/2.
            for kv_block_start, inner_iter_args in range(
                fx.Index(_f_col0),
                fx.Index(_f_col0 + _n_f * _BN_I32),
                BLOCK_N_OUT,
                init=init_args,
            ):
                # The successor of the last full tile is the first masked one,
                # which is only the adjacent tile when the left run is empty.
                _nxt = fx.Index(
                    fx.Int32(kv_block_start) + _BN_I32
                    if fx.Int32(kv_block_start) + _BN_I32 < _f_col0 + _n_f * _BN_I32
                    else _m_col0
                )
                loop_results = yield kv_loop_body(
                    kv_block_start, inner_iter_args, False, next_kv_start=_nxt
                )

            def _masked_col(i_idx):
                """Tile column for masked iteration i: the left run, then the
                right one. Discontinuous at the seam, which is exactly why the
                prefetch has to go through this same map."""
                _i = fx.Int32(i_idx)
                return (
                    _l_col0 + _i * _BN_I32
                    if _i < _n_l
                    else _r_col0 + (_i - _n_l) * _BN_I32
                )

            for _mi, inner_iter_args in range(
                fx.Index(0), _n_masked, 1, init=loop_results
            ):
                loop_results = yield kv_loop_body(
                    fx.Index(_masked_col(_mi)),
                    inner_iter_args,
                    True,
                    next_kv_start=fx.Index(_masked_col(fx.Int32(_mi) + fx.Int32(1))),
                )
        else:
            # Region 1: tiles that are wholly live -- no masking emitted at all.
            for kv_block_start, inner_iter_args in range(
                fx.Index(0), _full_end, BLOCK_N_OUT, init=init_args
            ):
                loop_results = yield kv_loop_body(
                    kv_block_start, inner_iter_args, False
                )

            # Region 2: the tail, where columns can be past seqlen_k or past the
            # causal diagonal.
            for kv_block_start, inner_iter_args in range(
                _full_end, kv_upper, BLOCK_N_OUT, init=loop_results
            ):
                loop_results = yield kv_loop_body(kv_block_start, inner_iter_args, True)

        # ---- Modification 3 (hipDNN vendoring): attention sink ----
        # gpt-oss's sink: one logit per query head that joins every row's softmax
        # as a key with no value. In this kernel's units (m is the scaled base-2
        # max) it is sink * log2(e). The row's max becomes m_top = max(m, sink2),
        # its sum l * 2^(m - m_top) + 2^(sink2 - m_top), and O is rescaled by
        # 2^(m - m_top) over that sum. A row no key reached (l == 0) keeps only
        # the sink: O = 0 and LSE = sink. The rewritten m and l feed the LSE and
        # the normalisation below unchanged; `_o_keep` carries the O rescale.
        _f32_ty_sink = ir.F32Type.get()
        _o_keep = [None] * ROW_SUBTILES
        if const_expr(HAS_SINK):
            _sink_raw = fx.Float32(
                fx.as_dsl_value(
                    _pointer_load(
                        elem_type,
                        buffer_ops.get_element_ptr(
                            fmha.pointer_to_llvm_ptr(SINK),
                            fx.Int64(head_q) * fx.Int64(sink_stride),
                            elem_type=elem_type,
                        ),
                    )
                ).to(fx.Float32)
            )
            _sink2 = fastmath.mul(_sink_raw, fx.Float32(_LOG2E))
            loop_results = list(loop_results)
            for qt in range_constexpr(ROW_SUBTILES):
                _m = fx.Float32(loop_results[2 * qt])
                _l = fx.Float32(loop_results[2 * qt + 1])
                _has = _l > fx.Float32(0.0)
                _m_top = _has.select(fastmath.max(_m, _sink2), _sink2)
                _keep = _has.select(
                    fx.Float32(
                        rocdl.exp2(
                            _f32_ty_sink, fx.as_ir_value(fastmath.sub(_m, _m_top))
                        )
                    ),
                    fx.Float32(0.0),
                )
                _l_tot = fastmath.add(
                    fastmath.mul(_l, _keep),
                    fx.Float32(
                        rocdl.exp2(
                            _f32_ty_sink, fx.as_ir_value(fastmath.sub(_sink2, _m_top))
                        )
                    ),
                )
                loop_results[2 * qt] = fx.as_ir_value(_m_top)
                loop_results[2 * qt + 1] = fx.as_ir_value(_l_tot)
                _o_keep[qt] = _keep

        # ---- logsumexp ----
        # LSE = (m + log2(l)) * ln2, with m in the base-2 scaled domain -- which
        # is exactly the convention the scaled-m softmax establishes, so nothing extra
        # is needed there. `rocdl.log` is v_log_f32, i.e. base 2.
        #
        # One value per (batch, head, q_row). A lane's m/l belong to its own
        # q_row (lane16), replicated across the klane halves by the shuffle_xor
        # reduction and across SHARDS by the cross-shard reduction, so exactly
        # one lane per row must store: klane 0 of shard 0.
        #
        # Layout is AOTriton's single branch-free formula
        #   offset = (b * H + h) * S + s
        # with (b=batch, s=0, S=Max_seqlen_q) giving (B*H, S). Varlen will pass
        # (b=0, s=cu_seqlens_q_start, S=total) for (H, TotalS) without changing
        # anything here -- that is the point of computing the base on the host.
        # Conditions are combined with `&`, never Python `and`/`not`: those
        # call `__bool__` on the MLIR value and are resolved at trace time,
        # which silently folded this whole block away on the first attempt.
        # `&` on an `fx.Boolean` is the bitwise op and evaluates neither side
        # at trace time.
        _f32_ty = ir.F32Type.get()
        _l_valid = fx.Int64(fx.ptrtoint(LSE)) != fx.Int64(0)
        _lse_writer = _l_valid & (klane == fx.Index(0)) & (shard_id == fx.Index(0))
        # Everything -- the log2, the scale, the address -- lives inside the
        # guard. Hoisting it out cost 8% at BLOCK_DMODEL 256 non-causal even though
        # the store itself was still predicated: the values stay live across
        # the epilogue and lengthen it for every wave, including the ones that
        # never store and the whole kernel when L is null.
        for qt in range_constexpr(ROW_SUBTILES):
            _do_store = _lse_writer & q_in_bounds_all[qt]
            if _do_store:
                _m = loop_results[2 * qt]
                _l = loop_results[2 * qt + 1]
                _lse = fastmath.mul(
                    fastmath.add(_m, rocdl.log(_f32_ty, fx.as_ir_value(_l))),
                    fx.Float32(_LN2),
                )
                # A row with no live keys gets +inf, not -inf: the backward
                # pass subtracts LSE from qk, so +inf makes exp(qk - inf)
                # zero for exactly the rows that must contribute nothing.
                # l is bit-exact 0 there, so test the bit pattern; integer
                # compares lower predictably.
                # Modification 2 (hipDNN vendoring): -inf for such a row, the
                # hipDNN / cuDNN convention, where AOTriton's backward wants +inf.
                _lse = (
                    fx.Float32(_lse)
                    if fmha.bitcast_i32(fx.Float32(_l)) != fx.Int32(0)
                    else fx.Float32(float("-inf"))
                )
                # LSE_LAYOUT, VarlenBits bits 17:16. The *inputs* are Q's
                # decode -- batch, row offset, length -- so the two layouts
                # are the same indices arranged two ways, not two features
                # (sdpa-varlen-plan.md section 3.2).
                #
                #   _HT  (H, T)   AOTriton's, and the default: T contiguous
                #   _TH  (T, H)   Transformer Engine's:         H contiguous
                # Compact in both layouts, so the head pitch is exactly
                # num_head_q and the token pitch is the decode's `tokens`.
                _row = _q_row_off_v + q_rows[qt]
                _lse_off, _ = fmha.lse_row_addressing(
                    varlen_bits, _q_batch_v, head_q, num_head_q, lse_tokens, _row
                )
                _pointer_store(
                    _lse,
                    buffer_ops.get_element_ptr(
                        fmha.pointer_to_llvm_ptr(LSE),
                        fx.Int64(_lse_off),
                        elem_type=_f32_ty,
                    ),
                )

        # ---- Normalize and store O ----
        # O's aperture: same columns as V, same rows as Q, and never staged.
        o_ap = fmha.Aperture(vo_cols, rows=q_ap.rows)

        def write_o(row, col, val):
            _store_global_half(o_ptr, o_tbase(start_q), o_toff(row, col), val)

        for qt in range_constexpr(ROW_SUBTILES):
            l_final = loop_results[2 * qt + 1]
            # A row can legitimately see *no* keys: bottom-right causal with
            # seqlen_q > seqlen_k leaves the leading seqlen_q - seqlen_k rows
            # fully masked, and bias or a sliding window will do the same.
            # Then l is exactly 0, 1/l is +inf, and o_acc * inf is NaN even
            # though o_acc is exactly 0.
            #
            # Clamp rather than branch: for a live row l >= 1 always, since
            # the running max contributes exp2(0) = 1, so this is a no-op
            # there.
            _l_safe = fastmath.max(l_final, fx.Float32(1e-30))
            inv_l = fastmath.div(c_one_f, _l_safe)
            if const_expr(HAS_SINK):
                inv_l = fx.as_ir_value(fastmath.mul(fx.Float32(inv_l), _o_keep[qt]))
            if const_expr(ENABLE_DROPOUT):
                # `1/(1-p)` folds into the existing `1/l` rather than becoming
                # a per-element multiply: it is uniform across the tile, and
                # the dropped entries are already zero, so scaling the whole
                # accumulator is equivalent and costs one scalar.
                #
                # `l` is deliberately *not* scaled -- it is the undropped sum,
                # and the logsumexp written below is the undropped one, which
                # is what the backward pass needs.
                inv_l = fx.as_ir_value(fastmath.mul(fx.Float32(inv_l), dropout_scale))
            inv_l_vec = (
                Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()
            )
            if q_in_bounds_all[qt]:
                for _oi in range_constexpr(O_ACCS):
                    vc, dc = _oi // D_CHUNKS, _oi % D_CHUNKS
                    o_norm_vec = fastmath.mul(
                        loop_results[_ML + qt * O_ACCS + _oi], inv_l_vec
                    )
                    o_trunc = Vec(o_norm_vec).to(elem_dtype).ir_value()
                    d_col = shard_vo_off + fx.Index(dc * D_CHUNK) + klane * 8
                    if const_expr(vc):
                        d_col = fx.Index(vc * VO_CHUNK_COLS) + d_col
                    if const_expr(D_OFFSET):
                        d_col = fx.Index(D_OFFSET) + d_col
                    fmha.write_v8(o_ap, write_o, q_rows_in_tile[qt], d_col, o_trunc)

    @flyc.jit
    def launch_flash_attn_aiw(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        batch_size: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        philox_seed_output: fx.Pointer,
        philox_offset_output: fx.Pointer,
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
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        SINK: fx.Pointer,
        sink_stride: fx.Int64,
        stride_b_seq_k: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()

        nseq_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        # Grid Q extent keys on Max_seqlen_q: under varlen there is no single
        # seqlen_q, so every sequence gets the longest one's worth of
        # workgroups and the short ones exit empty (plan section 6).
        sl_idx = fx.Index(max_seqlen_q)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M

        # Strides come from the caller, read off the real tensors -- never
        # derived here from num_heads/BLOCK_DMODEL. The shape does not determine
        # the layout (plan1 section 0), and K/V need not share Q's layout at
        # all. Axis 3 is D, contiguous by contract, so it is not passed.
        # Always forwarded: with STRIDES_CONSTEXPR the kernel simply does not
        # read them, which is what keeps the two arms directly comparable.
        launcher = flash_attn_func_aiw_kernel(
            Q,
            K,
            V,
            B,
            O,
            LSE,
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
            philox_seed_output,
            philox_offset_output,
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
            stride_o_batch,
            stride_o_head,
            stride_o_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            SINK,
            sink_stride,
            stride_b_seq_k,
        )

        if const_expr(WAVES_PER_EU is not None):
            _wpe = int(WAVES_PER_EU)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.WAVES_PER_EU"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        if const_expr(FLAT_WORK_GROUP_SIZE is not None):
            _fwgs = int(FLAT_WORK_GROUP_SIZE)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.FLAT_WORK_GROUP_SIZE"] = flat_wg_attr

        passthrough_entries = []
        # The default GCN scheduler sinks every LDS load next to its consuming
        # WMMA and funnels them all through one VGPR quad, so SIInsertWaitcnts
        # emits `s_wait_dscnt 0x0` between each load and use and the GEMMs run
        # with no LDS latency hiding. max-memory-clause trades VGPRs for keeping
        # several loads in flight.
        if const_expr(SCHED_STRATEGY):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("amdgpu-sched-strategy"),
                        ir.StringAttr.get(SCHED_STRATEGY),
                    ]
                )
            )
        if const_expr(DENORMALS_ARE_ZERO):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            if const_expr(FP_MODE == "fast"):
                passthrough_entries.append(
                    ir.ArrayAttr.get(
                        [
                            ir.StringAttr.get("no-nans-fp-math"),
                            ir.StringAttr.get("true"),
                        ]
                    )
                )
                passthrough_entries.append(
                    ir.ArrayAttr.get(
                        [ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")]
                    )
                )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(fx.Index(num_head_q), num_q_tiles, nseq_idx),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _fmha_compile_hints = {
        "FAST_FP_MATH": FAST_FP_MATH,
        "UNSAFE_FP_MATH": UNSAFE_FP_MATH,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    # Causal alignment is expressed as a *sentinel* window, resolved in the
    # kernel against each sequence's own lengths (`parse_window` in the
    # prologue). The host does not resolve it.
    #
    # It used to: `_CAUSAL_WINDOW` mapped causal_type 1/2 to literal bounds
    # using the single (seqlen_q, seqlen_k) pair passed to the launcher. That
    # is correct only when there is one such pair. Under varlen it silently
    # gave bottom-right the batch-wide `Max_seqlen_k - Max_seqlen_q` for every
    # sequence, which is wrong for all but the longest -- and invisible to any
    # test whose sequences share a uniform k-q difference, since then the two
    # numbers coincide.
    #
    # Resolving in one place removes the class of bug rather than the
    # instance.

    # ---- VarlenBits, sdpa-varlen-plan.md section 2 ----
    #
    # One byte per side, decoded by the same kernel-side function twice, plus
    # the LSE layout in byte 2. `0` is BHSD / MAX / IMPLIED on both sides with
    # an (H, T) logsumexp -- the conventional dense case, and the default.
    # `_HT` is AOTriton's and this kernel's default: shape (H, T), T
    # contiguous. `_TH` is Transformer Engine's (T, H).

    # 0x02

    def _bias_args(bias, like):
        """(pointer, stride_b_batch, stride_b_head, stride_b_seq_q) for a (B, H, Sq, Sk) bias.

        The last axis is the KV column and must be contiguous, exactly as the
        D axis is for Q/K/V/O -- the kernel loads eight adjacent columns per
        accumulator group in one v8.
        """
        if not BIAS_TYPE:
            return abi.NULL_PTR, 0, 0, 0, 0
        if bias is None:
            raise ValueError("this build has BIAS_TYPE=1 and requires bias=")
        if bias.dim() != 4:
            raise ValueError(
                f"bias must be rank 4 (B, H, Sq, Sk), got {tuple(bias.shape)}"
            )
        # Modification 5: the dtype is checked (an f32 mask passed to a build that
        # reads Q's dtype would be misread, not refused), and a strided key axis
        # needs a bias_key_strided build.
        _f32 = str(bias.dtype) == "torch.float32"
        if not (_f32 if BIAS_F32 else bias.dtype == like.dtype):
            raise ValueError(
                f"bias dtype {bias.dtype}: this build reads "
                f"{'f32' if BIAS_F32 else like.dtype}"
            )
        if bias.stride(3) != 1 and not BIAS_KSTRIDED:
            raise ValueError(
                f"bias must have a contiguous last (Sk) dimension, got "
                f"stride(3)={bias.stride(3)}; build with bias_key_strided=True"
            )
        return (
            abi.ptr_arg(bias),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
        )

    launch_flash_attn_aiw.compile_hints = dict(_fmha_compile_hints)

    def _args(
        Q,
        K,
        V,
        O,  # noqa: E741
        batch_size,
        seqlen_q,
        seqlen_k=None,
        num_seqlens=0,
        scale=None,
        stream=None,
        lse=None,
        window=None,
        varlen=None,
        bias=None,
        dropout_p=None,
        philox_seed=0,
        philox_offset1=None,
        philox_offset2=0,
        philox_seed_output=None,
        philox_offset_output=None,
        sink=None,
    ):  # noqa: E741
        """Every kernel argument but the stream, in launch order.

        `_launch` and `_compile` were 62 identical lines out of 66, differing
        only in whether the tail is `run_compiled` or `flyc.compile` and how
        the stream is spelled. dQ already had this shape; the forward did not.
        """
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = abi.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("O", O)], q_heads=("O",)
        )
        _lse_p = abi.lse_args(lse, seqlen_q, varlen, meta[0])
        _wl, _wr = abi.resolve_window(
            CAUSAL_TYPE, HOST_CAUSAL_TYPE, window, seqlen_q, seqlen_k
        )
        _vb, _sq0, _sq1, _sk0, _sk1, _mq, _mk = abi.varlen_args(
            STRIDES_CONSTEXPR, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens
        )
        _bp, _sb0, _sb1, _sb2, _sb3 = _bias_args(bias, Q)
        # `_hold` keeps a materialised philox scalar alive across the launch;
        # see `abi.u64_scalar`. Returned so the caller's frame owns it.
        _ps, _po1, _po2, _ip, _dsc, _hold = abi.dropout_args(
            ENABLE_DROPOUT,
            dropout_p,
            philox_seed,
            philox_offset1,
            philox_offset2,
            Q.device,
            stream,
        )
        _so, _oo = abi.dropout_outputs(
            ENABLE_DROPOUT, philox_seed_output, philox_offset_output
        )
        return (
            (
                # Q, K, V, B, <outputs> -- `ptrs` is (Q, K, V, O), so the
                # bias splices in ahead of O and the lse follows it as the
                # second output.
                *ptrs[:3],
                _bp,
                ptrs[3],
                _lse_p,
                _sq0,
                _sq1,
                _sk0,
                _sk1,
                _vb,
                batch_size,
                num_seqlens,
                _mq,
                _mk,
                _wl,
                _wr,
                _ps,
                _po1,
                _po2,
                _so,
                _oo,
                _ip,
                _dsc,
                *meta,
                abi.resolve_scale(Q, scale, PADDED_HEAD, sm_scale),
                *st,
                _sb0,
                _sb1,
                _sb2,
                abi.ptr_arg(sink) if sink is not None else abi.NULL_PTR,
                int(sink.stride(0)) if sink is not None else 0,
                _sb3,
            ),
            _hold,
            stream,
        )

    def _launch(*args, **kwargs):
        """Dispatch one forward pass. Signature is `_args`'s, which binds it."""
        packed, _hold, stream = _args(*args, **kwargs)
        abi.run_compiled(
            _COMPILED,
            launch_flash_attn_aiw,
            *packed,
            stream if stream is not None else fx.Stream(None),
        )

    def _compile(*args, **kwargs):
        """AOT-compile the same call `_launch` would dispatch."""
        packed, _hold, stream = _args(*args, **kwargs)
        return flyc.compile(launch_flash_attn_aiw, *packed, fx.Stream(stream))

    _launch.compile = _compile
    # Still attached to the launcher, for the callers and tests that reach them
    # that way -- but they now forward to `fmha_abi_gfx1201`, which is the one
    # copy of the wire spec. The old comment claimed they closed over
    # STRIDES_CONSTEXPR; only `varlen_args` does, and it takes it as an
    # argument now.
    _launch.varlen_bits = abi.varlen_bits
    _launch.varlen_compact = abi.varlen_compact
    _launch.varlen_padded = abi.varlen_padded
    _launch.varlen_strided = abi.varlen_strided
    _launch.varlen_seqused_k = abi.varlen_seqused_k
    return _launch


def build_flash_attn_func_aiw_module(**kwargs):
    """Keyword front end: name a problem, get the policy's schedule.

    Any `FmhaKnobs` field may be passed as a keyword to pin it; the rest are
    resolved by `resolve_knobs`. This is what keeps "the tuning module is the
    only producer of a schedule" true even for callers who never mention one.
    """
    meta_fields = {f.name for f in fields(FmhaInputMetadata)}
    knob_fields = {f.name for f in fields(FmhaKnobs)}
    unknown = set(kwargs) - meta_fields - knob_fields
    if unknown:
        raise TypeError(f"unknown build parameter(s): {sorted(unknown)}")
    # `resolve_knobs`, not `plan`: this front end takes a *compiled tile width*
    # and must keep rejecting anything off the ladder. Rounding a real head_dim
    # up is the interface's job, because only it also arranges the runtime
    # extent and the padded_head contract that make the rounding safe.
    meta = FmhaInputMetadata(**{k: v for k, v in kwargs.items() if k in meta_fields})
    overrides = FmhaKnobs(**{k: v for k, v in kwargs.items() if k in knob_fields})
    return build_flash_attn_func_aiw_module_primary(
        meta, resolve_knobs(meta, overrides)
    )
