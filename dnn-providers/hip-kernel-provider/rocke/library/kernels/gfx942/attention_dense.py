# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense flash-attention prefill kernel for gfx942 (CDNA3).

Port of the gfx950 dense prefill kernel (``kernels/gfx950/attention_dense.py``,
from PR #9480 / AICK-1663) to CDNA3. Tracked by **AICK-1664**.

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
    (CK ``transpose_vectors`` masks) into a transposed ``V_lds`` — the same vehicle
    already prototyped in ``kernels/gfx942/attention_tiled_2d.py``
    (``use_conflict_free_v_store``). It was long parked on a suspected
    store-mapping bug; that failure was re-tested and does NOT reproduce on
    ``develop`` (see the builder's ``IMPACT_LEDGER.md``), so P1 is a lift of a
    working vehicle, not a bug hunt.
  * **LDS / occupancy.** gfx942 has 64 KB LDS/CU (vs gfx950's 160 KB); occupancy /
    ``num_persistent`` / block sizing must be re-derived for the 228- and 304-CU
    gfx942 parts rather than inherited from the gfx950 tuning.

Problem category (drives the optimization order)
------------------------------------------------
The shipped gfx942 tiled-2D prefill kernel is **LDS-bank-conflict-bound ->
MFMA-starved** (NOT compute-bound, NOT HBM-bound): the V-read bank-conflict rate
dominates and leaves the MFMA pipe mostly idle, rather than sitting at the
HBM-bound roofline this problem should reach on this part. So conflict-free V and
softmax-VALU reduction rank first; compute-side scheduling levers that win on the
compute-bound gfx950 dense kernel (``s_setprio``, diagonal two-phase peel) are
PROVEN-NEGATIVE on gfx942 and are NOT ported. Measured conflict rates, pipe
utilisation and per-lever deltas are recorded outside the repo -- see the AICK-1664
plan and the protected results page.

Implementation status (see the AICK-1664 plan for the full ordered work list)
-----------------------------------------------------------------------------
  * P0  enablement + 32x32x8 atom + K-loop doubling ............ DONE (this file)
  * P1  conflict-free V (perm_b32 store-path transpose) ........ TODO
  * P2  exp2_fast + lazy_rescale ............................... TODO (portable)
  * P3  wide4 (WG=256) + K bank-pad retune + K single-buffer ... TODO
  * P4  persistent grid-stride + hkv_major decode .............. TODO
  * P5  diagonal masking (re-test only), partial-vmcnt prefetch  TODO

P0 is CORRECTNESS-FIRST: :func:`_build_attention_dense_p0` is non-pipelined (a
single LDS buffer) and reads V element-wise, so it is deliberately SLOWER than the
shipped ``attention_tiled_2d`` gfx942 kernel until P1-P3 land. It is validated
against an fp32 SDPA reference across the in-scope cohort; the perf levers layer on
top of it.

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
"""

from __future__ import annotations

from rocke.core.ir import IRBuilder, KernelDef, PtrType, BF16, F16, F32, I64
from rocke.helpers.attention import mfma_32x32x8_for_dtype

# The spec is arch-neutral (compile-time shape + tuning knobs); reuse it rather
# than fork the dataclass. gfx942-specific defaults live in the builder/dispatch.
from kernels.gfx950.attention_dense import (  # noqa: F401
    AttentionDenseSpec,
    _BLOCK_M,
    align_up,
)

# C-output lane maps: IDENTICAL between the 32x32x8 (gfx942) and 32x32x16 (gfx950)
# atoms (mfma_atom_catalog.md), so the softmax reductions + epilogue port verbatim.
from kernels.gfx942.attention_tiled_2d import _mfma_32x32_c_row, _mfma_32x32_c_col

LOG2E = 1.4426950408889634
_DTYPE_IR = {"bf16": BF16, "fp16": F16}

# P0 pipeline constants (mirror gfx950; NBUF=1 = non-pipelined for correctness-first).
_P0_BLOCK_M = 256  # query rows per CTA (8 wave64s)
# The kernel body tiles on _P0_BLOCK_M, attention_dense_grid sizes the launch grid
# from the shared _BLOCK_M, and supports_attention_dense's block_n divisibility
# check uses _P0_BLOCK_M. If those ever diverge the grid and the kernel disagree
# silently (rows written twice / never), so bind them at import instead of by comment.
if _P0_BLOCK_M != _BLOCK_M:  # not an `assert`: python -O would strip it
    raise ValueError(
        f"_P0_BLOCK_M ({_P0_BLOCK_M}) must match the shared _BLOCK_M ({_BLOCK_M}): "
        f"the launch grid and the kernel body would disagree silently, writing "
        f"query rows twice or not at all"
    )

# K-row bank-conflict pad, in elements. INHERITED from the gfx950 sibling and NOT
# re-derived for gfx942's bank geometry / LDS size -- that retune is a P3 item. Only
# applied when one K/V row is packed per async-DMA instruction; see
# _p0_lds_row_stride for why D64 cannot carry it.
_P0_LDS_PAD = 8

__all__ = [
    "AttentionDenseSpec",
    "supports_attention_dense",
    "build_attention_dense",
    "attention_dense_grid",
    "attention_dense_block",
    "attention_dense_signature",
    "p0_kernel_name",
    "run_attention_dense_torch",
]


def p0_kernel_name(spec: AttentionDenseSpec) -> str:
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
    """
    return f"{spec.kernel_name()}_gfx942_b{spec.batch}_wpe{spec.waves_per_eu}"


# In-scope for the gfx942 port (AICK-1664). supports_attention_dense rejects
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


def _p0_rows_per_instr(head_size: int) -> int:
    """K/V rows packed into one async-DMA instruction (1 at D128, 2 at D64).

    Single definition of the packing rule that BOTH the LDS row stride and the
    loader's addressing derive from; if those two ever disagree, the padded stride
    and the DMA disagree about row adjacency, which corrupts silently.
    """
    return _DMA_ELEMS_PER_INSTR // head_size


def _p0_lds_row_stride(head_size: int) -> int:
    """K_lds / V_lds row stride in ELEMENTS for one head size.

    D128 packs ONE row per async-DMA instruction (64 lanes x 2 elems = 128 elems =
    one D128 row), so the row can carry the bank-conflict pad. D64 packs TWO rows
    per instruction, which requires a contiguous UNPADDED stride -- a padded row
    would not be adjacent to the next one, so the single instruction could not
    cover both. D64 therefore takes LDS bank conflicts on the QK reads; widening
    that path is a P3 lever, not an oversight.

    Shared by the builder and :func:`supports_attention_dense` so the budget check
    cannot drift from the actual allocation.
    """
    return head_size + _P0_LDS_PAD if _p0_rows_per_instr(head_size) == 1 else head_size


def _p0_lds_bytes(spec: AttentionDenseSpec) -> int:
    """Total LDS footprint: K_lds + V_lds, each ``[1, block_n, row_stride]``.

    ``row_stride`` is :func:`_p0_lds_row_stride`; 2 bytes/element is exact for every
    dtype in ``_SUPPORTED_DTYPES`` (bf16/fp16) and must be revisited if a narrower or
    wider element type is added.
    """
    return 2 * spec.block_n * _p0_lds_row_stride(spec.head_size) * 2


def supports_attention_dense(
    spec: AttentionDenseSpec, *, arch: str = "gfx942"
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for one gfx942 dense-prefill config.

    This is the SINGLE source of truth for what :func:`build_attention_dense` can
    emit: every rejection the builder would make is made here first, so a True
    result implies the build succeeds. Dispatch gates on this, which lets an
    out-of-scope request fall through to another candidate instead of selecting
    this arm and failing at build time.

    In scope (AICK-1664 P0): gfx942, bf16/fp16, D64/D128, MHA/GQA including
    non-power-of-2 groups, causal or full, default grid, ``block_n`` dividing the
    256-row query tile, within the LDS budget and 32-bit addressing. Persistent is
    P4; varlen / ragged / sliding-window are later follow-ups.
    """
    if arch != "gfx942":
        return False, f"kernels.gfx942.attention_dense is gfx942-only (got {arch})"
    if spec.dtype not in _SUPPORTED_DTYPES:
        return (
            False,
            f"gfx942 attention_dense supports {_SUPPORTED_DTYPES}, got {spec.dtype}",
        )
    if spec.head_size not in _SUPPORTED_HEAD_SIZES:
        return False, (
            f"gfx942 attention_dense scope is D{list(_SUPPORTED_HEAD_SIZES)} "
            f"(D256 is AICK-1495/1496), got D{spec.head_size}"
        )
    if not isinstance(spec, AttentionDenseSpec):
        return False, f"spec must be an AttentionDenseSpec, got {type(spec).__name__}"
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

    # --- Mode scope. The P0 body implements the default-grid, uniform, dense
    # self-attention path only. Checked HERE and not only in the builder so that
    # support() and build() agree on exactly one set of specs.
    if spec.persistent:
        return False, "gfx942 attention_dense: persistent is P4 (AICK-1664)"
    if spec.varlen:
        return False, "gfx942 attention_dense P0: varlen not yet supported"
    if spec.ragged:
        return False, "gfx942 attention_dense P0: ragged not yet supported"
    if spec.sliding_window:
        return False, "gfx942 attention_dense P0: sliding_window not yet supported"

    # --- Tile geometry. The causal KV-loop clamp uses n_per = _P0_BLOCK_M //
    # block_n, a FLOOR: a block_n that does not divide the query tile silently drops
    # every key past the last whole sub-tile, and block_n > _P0_BLOCK_M makes n_per 0
    # -> zero-trip loop -> l == 0 -> rcp(0) -> NaN. Neither fails loudly, so reject.
    if _P0_BLOCK_M % spec.block_n != 0:
        return False, (
            f"block_n must divide the {_P0_BLOCK_M}-row query tile (got "
            f"block_n={spec.block_n}; the spec also requires block_n % 32 == 0, so "
            f"use 32, 64, 128 or 256). Load-bearing for causal=True, where "
            f"n_per = {_P0_BLOCK_M} // block_n floors and drops keys; enforced "
            f"unconditionally so the two grids cannot diverge by a knob"
        )

    # --- Wave/tile divisibility, mirrored from the builder so support() and build()
    # agree on exactly one set of specs (the module contract at the top of this file).
    # The condition below is ALSO enforced in _build_attention_dense_p0; without it
    # here, a spec support() accepted would die in the builder with a ValueError --
    # precisely the dispatch fall-through hole this gate exists to close.
    _waves = _P0_BLOCK_M // 32
    _rpi = _p0_rows_per_instr(spec.head_size)
    if spec.block_n % _waves != 0 or (spec.block_n // _waves) % _rpi != 0:
        return False, (
            f"block_n={spec.block_n} over {_waves} waves gives ROWS_PER_WAVE="
            f"{spec.block_n // _waves}, not a multiple of ROWS_PER_INSTR={_rpi} "
            f"(D={spec.head_size}); the async K/V DMA would skip rows"
        )

    # --- LDS budget. Without this, an over-budget tile reaches comgr and fails with
    # an opaque CODEGEN_BC_TO_RELOCATABLE abort instead of a structured reason.
    lds_bytes = _p0_lds_bytes(spec)
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
    spec: AttentionDenseSpec, *, arch: str = "gfx942"
) -> KernelDef:
    """Emit the gfx942 dense flash-attention prefill kernel for ``spec``.

    Delegates to :func:`_build_attention_dense_p0`, the correctness-first
    non-pipelined body (32x32x8 atom + K-loop doubling). Every scope restriction --
    including the modes deferred to P1-P4 -- lives in
    :func:`supports_attention_dense`, which is consulted here, so this function is a
    thin gate-plus-delegate and cannot reject a spec that ``supports`` accepted.

    :raises NotImplementedError: ``arch`` is not gfx942.
    :raises ValueError: ``spec`` is outside the supported set; the message carries
        :func:`supports_attention_dense`'s structured reason (out-of-scope dtype or
        head size, a mode deferred to a later phase, a ``block_n`` that does not
        divide the query tile, an over-budget LDS footprint, or an extent past the
        32-bit addressing limit).
    """
    if arch != "gfx942":
        raise NotImplementedError(
            f"kernels.gfx942.attention_dense is gfx942-only (got {arch})"
        )
    ok, why = supports_attention_dense(spec, arch=arch)
    if not ok:
        raise ValueError(f"unsupported gfx942 attention_dense spec: {why}")
    return _build_attention_dense_p0(spec)


def _build_attention_dense_p0(spec: AttentionDenseSpec) -> KernelDef:
    """P0 gfx942 dense prefill body: 32x32x8 atom + K-loop doubling.

    Correctness-first, non-pipelined (NBUF=1): per KV tile load→wait→QK→mask→
    softmax→PV, single LDS buffer, element-wise V read (bank-heavy but correct),
    simple online-softmax rescale. Perf levers (conflict-free V/cfvst=P1,
    exp2_fast+lazy=P2, wide/K-pad/pipeline=P3, persistent=P4) layer on top later.
    The transposed-QK architecture (S^T=K@Q^T so P feeds PV lane-locally) is kept
    from the gfx950 sibling — only the MFMA atom and K/V fragment widths change.
    """
    B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    Hq, Hkv, D = spec.num_query_heads, spec.num_kv_heads, spec.head_size
    causal = spec.causal
    dtype = _DTYPE_IR[spec.dtype]

    BLOCK_M = _P0_BLOCK_M
    WAVES = BLOCK_M // 32  # 8
    BN = spec.block_n

    N_SUB = BN // 32  # key 32-tiles per KV tile
    D_TILES = D // 32  # head-dim 32-tiles
    K_STEPS = D // 8  # QK K=8 doubled steps (was D//16 on gfx950)
    KK_STEPS = BN // 8  # PV K=8 doubled steps (was BN//16)
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D
    ROWS_PER_INSTR = _p0_rows_per_instr(D)  # 1 for D128, 2 for D64

    b = IRBuilder(p0_kernel_name(spec))
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)

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

    qb = b.block_id_x()
    hq = b.block_id_y()
    bt = b.block_id_z()
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

    # Padded for D128 (one row per DMA instr), unpadded for D64 (two rows per
    # instr need a contiguous stride) -- see _p0_lds_row_stride. Shared with the
    # LDS-budget check in supports_attention_dense so the two cannot drift.
    LDROW = _p0_lds_row_stride(D)
    K_lds = b.smem_alloc(dtype, [1, BN, LDROW], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [1, BN, LDROW], name_hint="Vlds")

    # Q packs (QK B-operand), scaled once by qk_scale so exp2(s) is direct.
    q_tok = b.add(q_tok0, lane_m)
    q_packs = []
    for ks in range(K_STEPS):
        col = b.add(b.const_i32(ks * 8), d_base)
        addr = b.add(b.add(q_base, b.mul(q_tok, b.const_i32(stride_q_tok))), col)
        raw = b.global_load_vN(q, addr, dtype, 4, align=8)
        elems = [
            b.cast_f32_to(b.fmul(b.cast_to_f32(b.vec_extract(raw, j)), qk_scale), dtype)
            for j in range(4)
        ]
        q_packs.append(b.vec_pack(elems, dtype))

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
    # same reasoning as the _P0_BLOCK_M guard at module scope. Mirrored in
    # supports_attention_dense so support() and build() cannot disagree.
    if BN % WAVES != 0 or ROWS_PER_WAVE % ROWS_PER_INSTR != 0:
        raise ValueError(
            f"block_n={BN} over {WAVES} waves gives ROWS_PER_WAVE={ROWS_PER_WAVE}, "
            f"not a multiple of ROWS_PER_INSTR={ROWS_PER_INSTR} (D={D}); the async "
            f"DMA would skip rows"
        )
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    k_rsrc = b.buffer_rsrc(k, b.const_i32(B * Skv * Hkv * D * 2))
    v_rsrc = b.buffer_rsrc(v, b.const_i32(B * Skv * Hkv * D * 2))

    def _async_load(rsrc, lds_base, tile_key0):
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
                row_lds_off = b.zext(b.mul(row0, b.const_i32(K_LDROW_BYTES)), I64)
                row_base = b.smem_ptr_add(lds_base, row_lds_off)
                gkey = b.add(b.add(tile_key0, row0), sub_row)
                voff = b.add(b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), col)
                b.async_buffer_load_lds_addr(
                    rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                )

    def load_tile(tile_idx):
        tk0 = b.mul(tile_idx, b.const_i32(BN))
        _async_load(k_rsrc, K_lds_addr, tk0)
        _async_load(v_rsrc, V_lds_addr, tk0)

    # ---- per-tile compute ----
    def do_qk():
        """S^T = K@Q^T via the 32x32x8 atom (K-doubled)."""
        s_reg = []
        for nsub in range(N_SUB):
            acc = b.zero_vec_f32(16)
            krow = b.add(b.const_i32(nsub * 32), lane_m)
            for ks in range(K_STEPS):
                col = b.add(b.const_i32(ks * 8), d_base)
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

    def relayout_p(p):
        """PV B-operand from lane-local P regs (no cross-lane): key=kk*8+lane_h*4+j
        maps to QK C-reg (nsub=kk//4, i=(kk%4)*4+j) — same lane_h split."""
        packs = []
        for kk in range(KK_STEPS):
            nsub = kk // 4
            base_i = (kk % 4) * 4
            elems = [b.cast_f32_to(p[nsub][base_i + j], dtype) for j in range(4)]
            packs.append(b.vec_pack(elems, dtype))
        return packs

    def read_v(dt, kk):
        """PV A-operand = V^T[dim, key], element-wise (P0). key=kk*8+lane_h*4+j,
        dim=dt*32+lane_m. Bank-heavy; cfvst conflict-free vehicle is P1."""
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
    iter_args = [("m", m0), ("l", l0)] + [(f"o{dt}", o0[dt]) for dt in range(D_TILES)]

    # elide_trailing_barrier=False: the trailing barrier is NOT an optimizable
    # rendezvous -- it is the WAR guard on the SINGLE K/V LDS buffer. The elide pass
    # (lower_llvm._lower_unrolled_for) targets body_ops[-2], which is exactly this
    # barrier's slot, and only misses it today because the op-name match is hardcoded
    # to "tile.sync" and unroll defaults False. Pin it so a future unroll=True (P3) or
    # a sync_lds_only->sync swap cannot silently delete it. Verified byte-identical
    # codegen with and without the flag today.
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
        alpha = b.exp2(b.fsub(m_i, m_new))

        p_vals = [
            [b.exp2(b.fsub(s[nsub][i], m_new)) for i in range(16)]
            for nsub in range(N_SUB)
        ]
        l_local = b.const_f32(0.0)
        for nsub in range(N_SUB):
            for i in range(16):
                l_local = b.fadd(l_local, p_vals[nsub][i])
        l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
        l_new = b.fadd(b.fmul(l_i, alpha), l_tile)

        o_acc = [
            b.vec_pack(
                [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)], F32
            )
            for dt in range(D_TILES)
        ]
        o_acc = do_pv(o_acc, relayout_p(p_vals))
        # Tile done; the next iteration's async DMA refills the SINGLE K/V buffer, so
        # every wave's LDS reads must have LANDED (not just issued) before any wave
        # starts writing. This MUST drain lgkmcnt and cannot be a bare s_barrier:
        # gfx90a+ set FeatureBackOffBarrier, so SIInsertWaitcnts skips the
        # conservative pre-barrier drain and places the lgkm wait at the ds_read's
        # CONSUMER -- which the scheduler is free to sink past the barrier. Measured
        # before the fix: ds_read_u16 x4 -> s_barrier -> s_waitcnt lgkmcnt(0).
        # sync_lds_only() emits the drain BEFORE the barrier (CK Tile block_sync_lds);
        # real LDS ops (mayLoad/mayStore) cannot be scheduled across the
        # side-effecting s_waitcnt, so the window is structurally closed. vmcnt is
        # deliberately NOT drained: it is already 0 here (the tile-start
        # s_waitcnt(vmcnt=0) drained this tile's DMA and NBUF=1 issues no new DMA
        # until the next iteration), so a full sync() would add a dead instruction.
        # NOTE: the gfx950 sibling is NOT a precedent for a bare barrier here. Its
        # NBUF=2 separates K (read j%2, written (j+1)%2) but NOT V: it reads
        # vbuf_prev=(j+1)%2 and writes pbuf=(j+1)%2 -- the SAME buffer, across a bare
        # s_barrier_bare(). It is correct there only by scheduler luck; tracked
        # separately, do not copy that idiom.
        b.sync_lds_only()
        b.scf_yield(m_new, l_new, *o_acc)

    res = loop.results
    l_i = res[1]
    o_acc = list(res[2 : 2 + D_TILES])

    # Epilogue: O[query,dim] = (P@V)/l, from the transposed C[dim,query] accumulator.
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
    b.ret()
    return b.kernel


# --- public geometry / ABI surface (arch-neutral; mirrors the gfx950 helpers) ---


def attention_dense_grid(spec: AttentionDenseSpec) -> tuple[int, int, int]:
    """Launch grid: persistent = 1-D grid of ``num_persistent`` CTAs; default =
    one CTA per (query-block, query-head, batch)."""
    if spec.persistent:
        return (spec.num_persistent, 1, 1)
    # ceil kept for parity with the gfx950 helper; on gfx942 it is always exact,
    # because ragged is rejected and the dataclass then enforces seqlen_q % 256 == 0.
    nqb = (spec.seqlen_q + _BLOCK_M - 1) // _BLOCK_M
    return (nqb, spec.num_query_heads, spec.batch)


def attention_dense_block(spec: AttentionDenseSpec) -> tuple[int, int, int]:
    """CTA block dims: ``num_waves`` wave64s."""
    return (spec.num_waves * 64, 1, 1)


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


def run_attention_dense_torch(*args, **kwargs):
    """End-to-end torch entry point for the gfx942 dense kernel.

    NOT YET IMPLEMENTED (AICK-1664). Will mirror
    ``kernels.gfx950.attention_dense.run_attention_dense_torch`` once
    :func:`build_attention_dense` emits a kernel.
    """
    raise NotImplementedError(
        "run_attention_dense_torch (gfx942) not yet implemented (AICK-1664)"
    )
