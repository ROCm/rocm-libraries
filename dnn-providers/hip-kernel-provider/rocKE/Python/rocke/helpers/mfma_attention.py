# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MFMA-tiled FMHA forward inner-body (production attention loop).

Replaces the warp-distributed scalar FMHA body
(:mod:`rocke.instances.common._fmha_warp_body`) with an MFMA-driven QK→
softmax→PV pipeline. One wave64 warp processes ``BLOCK_M = 16`` Q
rows per K-tile, ``BLOCK_K = 16`` K positions per iter. The MFMA
atom is ``mfma_f32_16x16x16_f16`` (or ``mfma_f32_16x16x32_f16`` for
the K-packed CDNA3 path). The QK + PV chain delivers 64 FLOPS /
cycle / lane vs the scalar inner's 2 FLOPS / cycle / lane.

What this helper does:

* **Q pre-load**: ``head_size / atom.k`` MFMA atom slabs of Q rows
  pre-loaded into per-lane f16 registers (no LDS staging for Q;
  it's constant over the K-loop so register-only is optimal).
* **K-tile loop** (``seqlen_k / BLOCK_K`` iterations):

  1. Load K tile into per-lane f16 registers.
  2. **QK MFMA chain**: ``head_size / atom.k`` MFMA invocations
     accumulating into a per-lane ``<4 x f32>`` score tile.
  3. Apply ``scale_log2`` and the spec's mask (causal / sliding
     window) via the per-row position check.
  4. **Per-row online softmax**: 16-lane butterfly reduce to get
     the row max, update ``m / l`` state, compute ``p = exp2(s -
     m_new)`` per row, rescale the previously-accumulated PV
     contribution by ``alpha = exp2(m_prev - m_new)``.
  5. **P operand staging**: cast f32 ``p`` to f16 and re-route via
     LDS so the PV MFMA gets it in the A-operand lane layout.
  6. **Load V tile** into per-lane f16 registers.
  7. **PV MFMA chain**: ``head_size / atom.n`` MFMA invocations
     accumulating into the per-lane f32 output.

* **Epilogue**: divide acc by ``l`` (per row), cast f32→f16, write
  to ``O`` in the atom's per-lane output layout.

What it leaves to the caller:

* Q / K / V / O global pointer & stride arithmetic (the kernel
  builder supplies these via callbacks).
* The grid layout (``q_tile_idx`` axis -- the caller's grid puts a
  CTA per ``(q_tile, head, batch)`` triple, where each q_tile
  spans ``BLOCK_M`` Q rows).
* The mask configuration (passed as a ``mask_mode`` string + a
  per-row query position callback).

The helper assumes ``head_size`` is a multiple of ``atom.k`` (16 for
the 16x16x16 atom, 32 for the K-packed 16x16x32 atom). All standard
head sizes (64, 128, 256) qualify.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..core.ir import F16, F32, BF16, IRBuilder, Value
from .atoms import MfmaAtom
from .attention import (
    apply_attention_mask,
    safe_inv_l,
    wave_reduce_stages,
)
from .distribution import (
    TileDistributionEncoding,
    block_tile_reduce_sync,
    make_static_distributed_tensor,
    make_static_tile_distribution,
)


# --- Distribution-driven softmax row reduce (CK Tile BlockReduce2dSync) -------
#
# The online-softmax row max / row sum fold each lane's per-row scalar across
# the 16 lanes that share that tile row. CK Tile expresses this as a
# ``block_tile_reduce_sync`` over a *reduce distribution*: a single keep-row Y
# (length 1) with the reduce axis collapsed into a lane-owned R level of length
# 16, derivative 1. ``_r_butterfly_plan`` then emits XOR masks ``1,2,4,8`` --
# byte-for-byte the historical ``wave_reduce_max/sum(lanes_per_row=16)``
# 4-stage butterfly (verified via the subset IR digest gate). This single
# encoding serves both the wave64 MFMA and the wave32 WMMA softmax: the lane
# butterfly stride is wave-size-independent (the ``wave_size`` arg only drives
# the cross-warp LDS stage, which a single-warp row reduce skips).
_SOFTMAX_ROW_REDUCE_ENC = TileDistributionEncoding(
    Rs=(16,),
    Hs=((1,),),
    Ps2RHs_major=((0,),),  # the single (lane) P feeds R (major 0)
    Ps2RHs_minor=((0,),),
    Ys2RHs_major=(1,),  # one keep-row Y on the M H-dim (length 1)
    Ys2RHs_minor=(0,),
)
_SOFTMAX_ROW_REDUCE_DIST = make_static_tile_distribution(_SOFTMAX_ROW_REDUCE_ENC)


def _softmax_row_reduce(b: IRBuilder, scalar: Value, *, combine: str) -> Value:
    """Reduce ``scalar`` across the 16 lanes sharing one tile row.

    Wraps the per-lane f32 ``scalar`` in a one-element
    :class:`StaticDistributedTensor` over :data:`_SOFTMAX_ROW_REDUCE_DIST`
    and folds it with :func:`block_tile_reduce_sync`. ``combine`` is
    ``"max"`` (row max) or ``"sum"`` (row sum). The emitted op stream is the
    same 4-stage XOR butterfly (masks ``1,2,4,8``) the legacy
    ``wave_reduce_max/sum(lanes_per_row=16)`` produced.
    """
    dt = make_static_distributed_tensor(_SOFTMAX_ROW_REDUCE_DIST, F32)
    dt.storage[0] = scalar
    block_tile_reduce_sync(b, dt, combine=combine)
    return dt.storage[0]


__all__ = [
    "MFMA_ATTN_BLOCK_M",
    "MFMA_ATTN_BLOCK_K",
    "mfma_attention_fwd_inner_body",
]


MFMA_ATTN_BLOCK_M = 16  # Q rows per CTA per K-tile
MFMA_ATTN_BLOCK_K = 16  # K positions per K-tile
_WAVE = 64


def _ir_type_for_dtype(dtype: str):
    if dtype in ("f16", "fp16"):
        return F16
    if dtype == "bf16":
        return BF16
    raise ValueError(f"mfma_attention currently supports f16/bf16; got {dtype!r}")


# Map MfmaAtom.dtype_in -> the catalog dtype key consumed by
# ``ArchTarget.mma`` (which normalises through ``normalize_dtype``).
_ATOM_DTYPE_TO_CATALOG = {
    "f16": "f16",
    "fp16": "f16",
    "bf16": "bf16",
    "fp8e4m3": "fp8",
    "bf8e5m2": "bf8",
    "fp4": "fp4",
    "fp6": "fp6",
}


def _validate_attention_atom(atom: "MfmaAtom", arch: str) -> None:
    """Reject an attention MFMA atom that is not in ``arch``'s catalog.

    Sources the legal atom set from :class:`rocke.core.arch.ArchTarget`
    (the SSOT in ``core/arch/data/arch_specs.json``) so the QK/PV atom
    selected by this helper is guaranteed to exist on the target before
    it reaches comgr. On the default ``arch="gfx950"`` every atom this
    helper can select is in the catalog, so this is a pure guard with no
    effect on the emitted IR.
    """
    from ..core.arch import ArchTarget

    cat_dtype = _ATOM_DTYPE_TO_CATALOG.get(atom.dtype_in, atom.dtype_in)
    target = ArchTarget.from_gfx(arch)
    if not target.mma.has_shape(
        a_dtype=cat_dtype,
        b_dtype=cat_dtype,
        c_dtype="fp32",
        m=atom.m,
        n=atom.n,
        k=atom.k,
    ):
        raise ValueError(
            f"mfma_attention: atom {atom.dtype_in} "
            f"{atom.m}x{atom.n}x{atom.k} (op_id {atom.name}) is not in the "
            f"{arch} MMA catalog; this kernel config is not legal on {arch}"
        )


def _load_kv_dequant_packed(
    b: IRBuilder,
    *,
    src: Value,
    addr: Value,
    n_elems: int,
    kv_dtype_eff: str,
    kv_dtype_ir,
    out_dtype_ir,
) -> Value:
    """Packed FP8 / BF8 K (or V) load with packed dequant.

    This is the P24 hoist: instead of issuing ``n_elems`` scalar
    ``global_load`` + ``cvt_fp8_to_f32`` calls (the original shape),
    issue ONE vector ``global_load_vN(.., n_elems)`` and dequant
    via ``cvt_pk_f32_fp8x4`` / ``cvt_pk_f32_bf8x4`` (one packed cvt
    per 4 elements). Then fptrunc to ``out_dtype_ir`` element-wise
    via :meth:`IRBuilder.vec_cast_f32_to`.

    For the standard ``a_per_lane = 4`` (f16-fallback fp8 path) this
    halves the vmem transactions and replaces 4 scalar cvts with 1
    packed cvt. For ``a_per_lane = 8`` (the future native fp8 atom
    path, P26) it cuts 8x → 1x vmem and 8x → 2x cvts.

    ``kv_dtype_eff`` is the canonical element name (``"fp8e4m3"`` /
    ``"bf8e5m2"``); ``kv_dtype_ir`` is the matching IR type;
    ``out_dtype_ir`` is the f16 / bf16 the MFMA atom consumes.
    """
    if n_elems % 4 != 0 or n_elems == 0:
        # Fall back to scalar dequant for non-multiples of 4. This
        # path only fires on adversarial atoms; the standard atoms
        # all use a_per_lane in {4, 8}.
        out = b.zero_vec(out_dtype_ir, n_elems)
        for j in range(n_elems):
            raw = b.global_load(src, b.add(addr, b.const_i32(j)), kv_dtype_ir, align=1)
            f32_v = (
                b.cvt_fp8_to_f32(raw)
                if kv_dtype_eff == "fp8e4m3"
                else b.cvt_bf8_to_f32(raw)
            )
            out = b.vec_insert(out, b.cast_f32_to(f32_v, out_dtype_ir), j)
        return out

    # One vmem transaction for the whole chunk.
    pk_vec = b.global_load_vN(src, addr, kv_dtype_ir, n_elems, align=n_elems)
    # Split into <4 x byte> halves, dequant each via the packed cvt,
    # concat, then fptrunc the <n_elems x f32> to <n_elems x f16/bf16>.
    f32_parts = []
    for grp in range(n_elems // 4):
        # Build a <4 x kv_dtype_ir> vector from positions
        # [grp*4 .. grp*4 + 3] of the loaded byte vector.
        chunk = b.zero_vec(kv_dtype_ir, 4)
        for j in range(4):
            scalar = b.vec_extract(pk_vec, grp * 4 + j)
            chunk = b.vec_insert(chunk, scalar, j)
        if kv_dtype_eff == "fp8e4m3":
            f32_chunk = b.cvt_pk_f32_fp8x4(chunk)
        else:
            f32_chunk = b.cvt_pk_f32_bf8x4(chunk)
        f32_parts.append(f32_chunk)
    if len(f32_parts) == 1:
        f32_full = f32_parts[0]
    else:
        f32_full = f32_parts[0]
        for next_part in f32_parts[1:]:
            f32_full = b.vec_concat(f32_full, next_part)
    return b.vec_cast_f32_to(f32_full, out_dtype_ir)


def mfma_attention_fwd_inner_body(
    b: IRBuilder,
    *,
    Q: Value,
    K: Value,
    V: Value,
    O: Value,  # noqa: E741 - standard attention notation (Q,K,V,O)
    head_size: int,
    seqlen_k: Value,
    q_tile_base: Value,
    head_idx: Value,
    kv_head_idx: Value,
    q_pos_base: Optional[Value] = None,
    stride_q_token: Value,
    stride_q_head: Value,
    stride_k_token: Value,
    stride_k_head: Value,
    stride_v_token: Value,
    stride_v_head: Value,
    stride_o_token: Value,
    stride_o_head: Value,
    scale_log2: Value,
    dtype: str = "f16",
    mask_mode: str = "none",
    sliding_window: int = 0,
    causal_ctx_offset: Optional[Value] = None,
    k_token_offset_elems: Optional[Value] = None,
    v_token_offset_elems: Optional[Value] = None,
    k_row_base_fn: Optional[Callable[[IRBuilder, Value], Value]] = None,
    v_row_base_fn: Optional[Callable[[IRBuilder, Value], Value]] = None,
    k_tile_start: Optional[Value] = None,
    k_tile_stop: Optional[Value] = None,
    extra_score_transform: Optional[
        Callable[[IRBuilder, Value, Value, int], Value]
    ] = None,
    extra_mask_predicate: Optional[Callable[[IRBuilder, Value], Value]] = None,
    extra_skip_predicate: Optional[Callable[[IRBuilder, Value], Value]] = None,
    k_block_iter_fn: Optional[Callable[[IRBuilder, Value], Value]] = None,
    kv_dtype: Optional[str] = None,
    v_scale: Optional[Value] = None,
    use_wider_atom: bool = False,
    native_fp8_path: bool = False,
    use_async_kv: bool = False,
    codebook_ptr: Optional[Value] = None,
    wmma_v_lds_stage: bool = False,
    arch: str = "gfx950",
) -> None:
    """One MFMA-tiled QK→softmax→PV pass for a ``BLOCK_M``-row Q tile.

    The kernel must launch with ``block_size = 64`` (one wave64 warp
    per CTA). The CTA processes ``BLOCK_M = 16`` Q rows starting at
    ``q_tile_base``, accumulating attention from the full K span
    (``seqlen_k`` positions) in ``seqlen_k / BLOCK_K`` K-tiles.

    ``causal_ctx_offset`` (when ``mask_mode == "causal"``) is the
    offset added to each row's local position to get its
    causal-mask threshold (``k_idx <= q_pos + offset``). For
    self-attention pass ``b.const_i32(0)``; for cross-attention pass
    the cache length.

    ``k_token_offset_elems`` / ``v_token_offset_elems`` are added to
    the K / V row base addresses (for varlen / paged-KV layouts).

    ``k_row_base_fn`` / ``v_row_base_fn``: callbacks
    ``(b, k_row_idx) -> i32`` returning the linear element offset
    for one K / V row. Overrides the default dense addressing
    (``k_row_idx * stride + head_offset + token_offset``). Used by
    the paged-KV kernel where each ``k_row_idx`` indirects through
    a ``block_table`` to a physical block.

    ``k_tile_start`` / ``k_tile_stop``: when set, the K-tile loop
    runs in ``[k_tile_start, k_tile_stop)`` instead of ``[0,
    seqlen_k / BLOCK_K)``. Used by the split-KV decode segment
    kernel so a single CTA handles one K segment.

    ``extra_score_transform``: callback
    ``(b, score_log2_per_lane, k_tile_idx, row_in_atom) ->
    score_log2`` invoked after the QK reduction and ``scale_log2``
    multiply, before the mask. Used by sage attention to apply
    per-block Q + K scales.

    ``extra_mask_predicate``: callback ``(b, k_tile_idx) -> i1``
    returning a per-K-tile keep flag. When false, the whole K-tile
    is skipped (no MFMA, no V load, no PV). Used by block-sparse
    (jenga / VSA) attention to short-circuit non-attended K-blocks.

    ``kv_dtype``: K / V storage dtype when it differs from Q's
    ``dtype`` (e.g. K/V in fp8 while Q is f16). The helper does
    inline ``cvt_fp8_to_f32 → cast_f32_to_f16 → f16 MFMA`` dequant
    on the load path; the native fp8 MFMA atom (``mfma_f32_16x16x32_fp8``)
    is a v2 hoist.

    ``v_scale`` (optional f32): per-tensor V dequant scale. When set,
    the final accumulator is multiplied by ``v_scale`` at the
    epilogue (mathematically equivalent to scaling each V dequant
    output by ``v_scale``). Callers that need per-tensor ``k_scale``
    just pre-multiply ``scale_log2`` by ``k_scale`` before invoking
    the helper -- the QK MFMA result lands in the log2-space score,
    and a constant K-scale is absorbed cleanly into ``scale_log2``.
    """
    if head_size % MFMA_ATTN_BLOCK_M != 0:
        raise ValueError(
            f"mfma_attention head_size {head_size} must be a multiple of "
            f"{MFMA_ATTN_BLOCK_M}"
        )
    if dtype not in ("f16", "fp16", "bf16"):
        raise ValueError(f"mfma_attention dtype must be f16/bf16, got {dtype!r}")

    # Q dtype is the activation dtype; K / V dtype can be fp8e4m3 /
    # bf8e5m2 (when ``kv_dtype`` is set). The QK MFMA atom picks
    # ``f16 ⊗ f16 → f32``, ``bf16 ⊗ bf16 → f32`` (gfx940+ via
    # ``mfma_f32_16x16x16_bf16``) or ``fp8 ⊗ fp8 → f32`` based on K/V
    # dtype.
    #
    # P25: ``use_wider_atom=True`` swaps the 16x16x32 fp8 / bf8 atom
    # for the 32x32x16 hero. The softmax row-reduce branch in
    # ``helpers/attention.py`` already exposes ``warp_xor_reduce_*_32lane``
    # for 32-lane rows; the call site is responsible for matching its
    # row-reduce stage count to the atom's lane layout. For the f16 /
    # bf16 path this flag is a no-op (the 16x16x16 atom is already
    # the canonical small-tile shape).
    if kv_dtype is None or kv_dtype == dtype:
        if dtype == "bf16":
            atom = MfmaAtom.bf16_16x16x16()
        else:
            atom = MfmaAtom.f16_16x16x16()
        kv_dtype_eff = dtype
    elif kv_dtype == "fp8e4m3":
        atom = MfmaAtom.fp8_32x32x16() if use_wider_atom else MfmaAtom.fp8_16x16x32()
        kv_dtype_eff = "fp8e4m3"
    elif kv_dtype == "bf8e5m2":
        atom = MfmaAtom.bf8_32x32x16() if use_wider_atom else MfmaAtom.bf8_16x16x32()
        kv_dtype_eff = "bf8e5m2"
    else:
        raise ValueError(
            f"mfma_attention: unsupported kv_dtype {kv_dtype!r}; "
            "expected None / 'f16' / 'fp8e4m3' / 'bf8e5m2'"
        )
    if head_size % atom.k != 0:
        raise ValueError(
            f"head_size {head_size} must be a multiple of atom.k "
            f"{atom.k} for the selected atom"
        )

    # The fp8 KV path requires Q and K to share the same MFMA-input
    # dtype, so when ``kv_dtype`` is fp8/bf8, the kernel is responsible
    # for pre-casting Q to that dtype before this helper runs. The
    # helper itself uses ``kv_dtype_eff`` for both operands.
    dtype_ir = _ir_type_for_dtype(dtype if kv_dtype_eff == dtype else "f16")
    kv_dtype_ir = (
        dtype_ir
        if kv_dtype_eff == dtype
        else (
            F16
            if kv_dtype_eff in ("f16", "fp16")
            else (BF16 if kv_dtype_eff == "bf16" else dtype_ir)
        )
    )
    # P26: ``native_fp8_path=True`` keeps the fp8 atom selected above
    # and skips the f16 fallback dequant chain — Q must be pre-cast to
    # fp8 outside this helper, and the QK MFMA runs natively
    # ``mfma_f32_16x16x32_fp8``. P also gets quantised to fp8 inside
    # the K-loop via ``cvt_pk_fp8_f32x4`` (the cvt is already in IR).
    # When ``native_fp8_path=False`` (default), we dequantise K/V on
    # load and run the f16 MFMA — slower but smaller code.
    from ..core.ir import BF8E5M2, FP8E4M3

    if kv_dtype_eff != dtype and not native_fp8_path:
        # Fall back to the f16 atom; we'll dequantise K/V on load.
        atom = MfmaAtom.f16_16x16x16()
        dtype_ir = F16
        kv_dtype_ir = FP8E4M3 if kv_dtype_eff == "fp8e4m3" else BF8E5M2
    elif kv_dtype_eff != dtype and native_fp8_path:
        # Keep the fp8 atom; the input dtype must be fp8 for both
        # operands. Caller is responsible for the Q pre-cast; we set
        # ``dtype_ir`` to the fp8 IR type so the load path issues
        # fp8 vmem.
        dtype_ir = FP8E4M3 if kv_dtype_eff == "fp8e4m3" else BF8E5M2
        kv_dtype_ir = dtype_ir

    fp8_kv = kv_dtype_eff != dtype

    # --- Arch / wave dispatch (MMA contract) ----------------------------------
    # Resolve the target so the body can select the matmul *op* (MFMA on CDNA,
    # WMMA on the RDNA wave32 targets) from the same catalog the GEMM
    # unification uses. On a wave32 target (gfx1151) the QK/PV chain, the
    # online-softmax row reduction, and the P fragment re-layout are all driven
    # off the WMMA ``MmaOp`` layout maps via :func:`_wmma_attention_fwd_inner_body`
    # -- the wave32 attention analogue of the unified GEMM's WMMA branch. On a
    # wave64 CDNA target the body continues below unchanged (byte-identical to
    # the historical MFMA emission); the only contract touch on that path is
    # routing the matmul through the op's ``op_id`` (identical lowering to the
    # ISA-named MFMA call) and the wave64 reduction helper (identical XOR
    # butterfly).
    #
    # This dispatch precedes the MFMA-atom catalog guard below because the
    # wave32 path does not use the MFMA ``atom`` object at all (it resolves its
    # own WMMA op); validating the MFMA atom against the RDNA catalog would
    # falsely reject the legal WMMA config.
    from ..core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    wave_size = target.wave_size

    if wave_size == 32:
        # RDNA wave32 (WMMA). The fp8 / wider-atom / native-fp8 KV paths are
        # CDNA-only (no RDNA atom); reject them explicitly rather than emitting
        # an unbuildable kernel.
        if fp8_kv or use_wider_atom or native_fp8_path:
            raise ValueError(
                "wave32 (WMMA) attention supports f16/bf16 KV only; "
                "fp8 / wider-atom / native-fp8 paths are CDNA-only"
            )
        _wmma_attention_fwd_inner_body(
            b,
            Q=Q,
            K=K,
            V=V,
            O=O,
            head_size=head_size,
            seqlen_k=seqlen_k,
            q_tile_base=q_tile_base,
            head_idx=head_idx,
            kv_head_idx=kv_head_idx,
            q_pos_base=q_pos_base,
            stride_q_token=stride_q_token,
            stride_q_head=stride_q_head,
            stride_k_token=stride_k_token,
            stride_k_head=stride_k_head,
            stride_v_token=stride_v_token,
            stride_v_head=stride_v_head,
            stride_o_token=stride_o_token,
            stride_o_head=stride_o_head,
            scale_log2=scale_log2,
            dtype=dtype,
            mask_mode=mask_mode,
            sliding_window=sliding_window,
            causal_ctx_offset=causal_ctx_offset,
            k_token_offset_elems=k_token_offset_elems,
            v_token_offset_elems=v_token_offset_elems,
            k_row_base_fn=k_row_base_fn,
            v_row_base_fn=v_row_base_fn,
            k_tile_start=k_tile_start,
            k_tile_stop=k_tile_stop,
            extra_score_transform=extra_score_transform,
            extra_mask_predicate=extra_mask_predicate,
            extra_skip_predicate=extra_skip_predicate,
            k_block_iter_fn=k_block_iter_fn,
            v_scale=v_scale,
            v_lds_stage=wmma_v_lds_stage,
            arch=arch,
            target=target,
        )
        return

    # --- CDNA wave64 (MFMA) path ----------------------------------------------
    # Arch guard: the QK/PV MFMA atom selected above must be in the target's MMA
    # catalog. This is the machine-checkable hook that keeps gfx942 from
    # selecting a gfx950-only atom (wide f16/bf16 16x16x32 / 32x32x16, or
    # fp4/mx) -- comgr would otherwise HARD-CRASH with ``LLVM ERROR: Cannot
    # select intrinsic`` on the missing op. The default arch is gfx950, and
    # every atom this helper selects today exists on both gfx942 and gfx950, so
    # for the default the selected atom is unchanged. The guard emits no IR, so
    # it cannot perturb the byte-identical CDNA emission.
    _validate_attention_atom(atom, arch)

    # The matmul op_id sourced from the catalog; ``b.mma(op, ...)`` lowers
    # identically to the historical ``atom.emit`` (the ISA-named MFMA call), so
    # the emitted IR is byte-for-byte unchanged.
    qk_op = target.mma.by_op_id(atom.name)

    n_qk_atoms = head_size // atom.k
    n_pv_atoms = head_size // atom.n  # also == head_size / 16

    lane = b.thread_id_x()
    c16 = b.const_i32(16)
    m_in_atom = b.mod(lane, c16)  # 0..15 -- Q row within the BLOCK_M tile
    k_blk = b.div(lane, c16)  # 0..3  -- which 4-K-element slot
    c_a_per_lane = b.const_i32(atom.a_per_lane)
    k_lane_start = b.mul(k_blk, c_a_per_lane)

    k_off = k_token_offset_elems if k_token_offset_elems is not None else b.const_i32(0)
    v_off = v_token_offset_elems if v_token_offset_elems is not None else b.const_i32(0)

    # ---- Pre-load Q ----
    # For each QK atom along head_dim, lane t holds Q[q_tile_base +
    # m_in_atom, k_blk_atom*atom.k + k_blk*a_per_lane + 0..a_per_lane).
    q_row = b.add(q_tile_base, m_in_atom)
    q_addr_row_base = b.add(
        b.mul(q_row, stride_q_token),
        b.mul(head_idx, stride_q_head),
    )
    q_vecs = []
    for k_blk_atom in range(n_qk_atoms):
        d_start = b.add(
            b.mul(b.const_i32(k_blk_atom), b.const_i32(atom.k)),
            k_lane_start,
        )
        q_addr = b.add(q_addr_row_base, d_start)
        q_vecs.append(
            b.global_load_vN(
                Q,
                q_addr,
                dtype_ir,
                atom.a_per_lane,
                align=atom.a_per_lane * 2,
            )
        )

    # ---- LDS for P-operand staging ----
    # After softmax we need P in the A-operand lane layout for the PV
    # MFMA. The score tile lives in registers as 4 cells per lane (4
    # rows × 1 col). To redistribute for PV we round-trip through
    # LDS: store each lane's 4 cells into P_lds at (row, col), sync,
    # load back in the new layout.
    P_lds = b.smem_alloc(
        dtype_ir,
        [MFMA_ATTN_BLOCK_M, MFMA_ATTN_BLOCK_K],
        name_hint="Pmfma",
    )

    # ---- Online softmax + PV accumulator iter_args ----
    # Each lane has 4 rows worth of (m, l) state: rows m_blk*4 + i for
    # i in 0..3, where m_blk = lane / 16. We carry m_r, l_r per row
    # slot through the K-loop.
    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)
    acc_zero = b.zero_vec_f32(atom.c_per_lane)

    iter_args = []
    for r in range(atom.c_per_lane):
        iter_args.append((f"m{r}", neg_inf))
        iter_args.append((f"l{r}", zero_f))
    for n in range(n_pv_atoms):
        iter_args.append((f"acc{n}", acc_zero))

    c_block_k = b.const_i32(MFMA_ATTN_BLOCK_K)
    loop_start = k_tile_start if k_tile_start is not None else b.const_i32(0)
    loop_stop = k_tile_stop if k_tile_stop is not None else b.div(seqlen_k, c_block_k)

    kloop = b.scf_for_iter(
        loop_start,
        loop_stop,
        b.const_i32(1),
        iter_args=iter_args,
        iv_name="kt",
    )
    with kloop as (kt, state_vals):
        # Unpack state.
        ms = [state_vals[2 * r] for r in range(atom.c_per_lane)]
        ls = [state_vals[2 * r + 1] for r in range(atom.c_per_lane)]
        accs = list(state_vals[2 * atom.c_per_lane :])

        # P29: k_block_iter_fn maps the linear iter index to a real
        # K-tile index via a caller-supplied LUT. ``effective_kt`` is
        # the K-tile index every downstream computation uses; the
        # raw ``kt`` is just the loop counter. This lets VSA and
        # other block-sparse callers iterate only the attended K-blocks
        # in dense order rather than masking every K-tile.
        if k_block_iter_fn is not None:
            effective_kt = k_block_iter_fn(b, kt)
        else:
            effective_kt = kt

        # K row base for this lane: K[k_tile_base + n_in_atom, ...]
        # where n_in_atom = m_in_atom (same lane decoded for both A
        # and B operands in the QK MFMA).
        k_tile_base = b.mul(effective_kt, c_block_k)
        k_row_for_lane = b.add(k_tile_base, m_in_atom)  # k position for THIS lane

        # Per-K-tile mask predicate (block-sparse / VSA): when false, the
        # whole tile is skipped -- no K/V loads, no MFMA, no PV. The
        # iter_args carry through unchanged so the softmax state stays
        # consistent across skipped tiles.
        if extra_mask_predicate is not None:
            keep_tile = extra_mask_predicate(b, kt)
        else:
            keep_tile = None
        # P28: ``extra_skip_predicate`` short-circuits the entire K-tile
        # body — when False, no K/V loads, no MFMA, no PV fire (the
        # softmax-mask path above keeps the loads but forces score to
        # -inf, paying bandwidth + compute for masked tiles). Combined
        # with ``keep_tile`` so callers can use both.
        if extra_skip_predicate is not None:
            skip_mask = extra_skip_predicate(b, kt)
            keep_tile = (
                b.land(keep_tile, skip_mask) if keep_tile is not None else skip_mask
            )

        # Compute the K row base for this lane. Default is dense
        # ``k_row * stride_k_token + kv_head * stride_k_head + k_off``;
        # callbacks override for paged-KV / varlen.
        if k_row_base_fn is not None:
            k_addr_row_base = k_row_base_fn(b, k_row_for_lane)
        else:
            k_addr_row_base = b.add(
                b.add(
                    b.mul(k_row_for_lane, stride_k_token),
                    b.mul(kv_head_idx, stride_k_head),
                ),
                k_off,
            )

        # ---- QK MFMA chain ----
        # score = sum over qk_atoms of MFMA(Q[atom], K[atom], score)
        # Per-lane <4 x f32> (4 row-cells, 1 col-cell).
        # When K/V are fp8 / bf8, we load the bytes packed and dequant
        # via ``cvt_pk_f32_fp8x4`` / ``cvt_pk_f32_bf8x4`` (P24): one
        # vmem load + 1-2 packed cvts vs ``a_per_lane`` scalar loads
        # + ``a_per_lane`` cvts in the original shape. Same FLOPs in
        # the MFMA but ~8x → 1x vmem transactions and ~8x → 2x
        # cvt-class instructions per K-tile per atom on the fp8 path.
        score = atom.zero_acc(b)
        for k_blk_atom in range(n_qk_atoms):
            d_start = b.add(
                b.mul(b.const_i32(k_blk_atom), b.const_i32(atom.k)),
                k_lane_start,
            )
            k_addr = b.add(k_addr_row_base, d_start)
            if fp8_kv:
                k_vec = _load_kv_dequant_packed(
                    b,
                    src=K,
                    addr=k_addr,
                    n_elems=atom.a_per_lane,
                    kv_dtype_eff=kv_dtype_eff,
                    kv_dtype_ir=kv_dtype_ir,
                    out_dtype_ir=dtype_ir,
                )
            else:
                k_vec = b.global_load_vN(
                    K,
                    k_addr,
                    dtype_ir,
                    atom.a_per_lane,
                    align=atom.a_per_lane * 2,
                )
            score = b.mma(qk_op, q_vecs[k_blk_atom], k_vec, score)

        # ---- Scale + mask + softmax row update ----
        # Each lane has 4 score cells at (m_blk*4 + i, n_in_atom).
        # m_blk = lane / 16, n_in_atom = lane % 16 = m_in_atom.
        m_blk = b.div(lane, c16)
        # Per-row max via 16-lane butterfly reduction (lanes in the same
        # m_blk reduce across cols 0..15).
        new_ms, new_ls, new_accs = [], [], list(accs)
        ps = []  # per-row scaled probabilities (per-lane f32)
        for r in range(atom.c_per_lane):
            s_r_f32 = b.vec_extract(score, r)
            s_r_scaled = b.fmul(s_r_f32, scale_log2)
            # Per-row position for the mask check + extra transforms.
            # ``q_pos_base`` (defaults to ``q_tile_base``) is the
            # position used by the mask predicate; it's distinct from
            # ``q_tile_base`` (which addresses Q rows in global memory)
            # so callers with a batch axis can pass the within-batch Q
            # position for the causal / sliding-window mask while
            # still using the global-batched Q address.
            q_pos_for_mask = q_pos_base if q_pos_base is not None else q_tile_base
            row_q_pos = b.add(
                b.add(q_pos_for_mask, b.mul(m_blk, b.const_i32(4))),
                b.const_i32(r),
            )
            k_col_pos = b.add(k_tile_base, m_in_atom)
            if extra_score_transform is not None:
                s_r_scaled = extra_score_transform(
                    b,
                    s_r_scaled,
                    kt,
                    r,
                )
            s_r_scaled = apply_attention_mask(
                b,
                s_r_scaled,
                mask_mode=mask_mode,
                k_idx=k_col_pos,
                query_pos=row_q_pos,
                sliding_window=sliding_window,
                context_len=causal_ctx_offset,
            )
            # If the whole K-tile is masked off (sparse-skip), force the
            # score to -inf so the softmax exponential collapses to 0.
            if keep_tile is not None:
                s_r_scaled = b.select(keep_tile, s_r_scaled, neg_inf)
            # 16-lane row-max reduce via the distribution-driven
            # ``block_tile_reduce_sync`` (CK Tile ``BlockReduce2dSync``). The
            # reduce distribution's lane-owned R level (length 16, derivative 1)
            # emits the exact 4-stage XOR butterfly (masks 1,2,4,8) the legacy
            # ``wave_reduce_max(lanes_per_row=16)`` produced.
            row_max = _softmax_row_reduce(b, s_r_scaled, combine="max")
            m_new_r = b.fmax(ms[r], row_max)
            alpha_r = b.exp2(b.fsub(ms[r], m_new_r))
            p_r = b.exp2(b.fsub(s_r_scaled, m_new_r))
            # Row-sum reduce of p.
            row_psum = _softmax_row_reduce(b, p_r, combine="sum")
            l_new_r = b.fadd(b.fmul(ls[r], alpha_r), row_psum)

            new_ms.append(m_new_r)
            new_ls.append(l_new_r)
            ps.append(p_r)
            # Rescale every PV accumulator slot for row r.
            for n in range(n_pv_atoms):
                # Accumulator slot n's row r is in vec slot r.
                old = b.vec_extract(new_accs[n], r)
                rescaled = b.fmul(old, alpha_r)
                new_accs[n] = b.vec_insert(new_accs[n], rescaled, r)

        # ---- P operand staging via LDS ----
        # Lane t holds p[m_blk*4+r, n_in_atom] for r in 0..3.
        # Write to P_lds[m_blk*4+r, n_in_atom] then sync.
        for r in range(atom.c_per_lane):
            p_row = b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(r))
            p_col = m_in_atom
            p_f16 = b.cast_f32_to(ps[r], dtype_ir)
            b.smem_store_vN(P_lds, [p_row, p_col], p_f16, 1)
        b.sync()

        # ---- PV MFMA chain ----
        # For each PV atom along the n (head_dim) axis, load:
        #   * P operand: lane t holds P[m_in_atom, k_blk*4 + 0..3]
        #     -- these are 4 P cols at row m_in_atom.
        #   * V operand: lane t holds V[k_tile_base + k_blk*4 + 0..3,
        #     n_blk_atom * atom.n + n_in_atom] -- 4 V values at col
        #     (n_blk_atom * 16 + n_in_atom).
        # Where n_in_atom = m_in_atom and k_blk*4..k_blk*4+3 are 4
        # K-positions within the BLOCK_K K-tile.
        for n_blk_atom in range(n_pv_atoms):
            # P operand load from LDS.
            p_a_vec = b.zero_vec(dtype_ir, atom.a_per_lane)
            for j in range(atom.a_per_lane):
                p_col_j = b.add(k_lane_start, b.const_i32(j))
                p_v = b.vec_extract(
                    b.smem_load_vN(P_lds, m_in_atom, p_col_j, dtype=dtype_ir, n=1),
                    0,
                )
                p_a_vec = b.vec_insert(p_a_vec, p_v, j)
            # V operand: per-lane scalar loads at strided V[k, n_col].
            v_col_in_hd = b.add(
                b.mul(b.const_i32(n_blk_atom), b.const_i32(atom.n)),
                m_in_atom,
            )
            v_a_vec = b.zero_vec(dtype_ir, atom.b_per_lane)
            for j in range(atom.b_per_lane):
                v_row_k = b.add(
                    k_tile_base,
                    b.add(k_lane_start, b.const_i32(j)),
                )
                if v_row_base_fn is not None:
                    v_addr_row_base = v_row_base_fn(b, v_row_k)
                else:
                    v_addr_row_base = b.add(
                        b.add(
                            b.mul(v_row_k, stride_v_token),
                            b.mul(kv_head_idx, stride_v_head),
                        ),
                        v_off,
                    )
                v_addr = b.add(v_addr_row_base, v_col_in_hd)
                if fp8_kv:
                    # The PV V load is per-row-strided (a different K-row
                    # per j) so it can't directly use the packed-dequant
                    # vmem transaction the K path uses (which is
                    # contiguous in head_dim). Future improvement: prefetch
                    # rows into LDS and read them via the packed cvt.
                    raw = b.global_load(V, v_addr, kv_dtype_ir, align=1)
                    f32_v = (
                        b.cvt_fp8_to_f32(raw)
                        if kv_dtype_eff == "fp8e4m3"
                        else b.cvt_bf8_to_f32(raw)
                    )
                    v_scalar = b.cast_f32_to(f32_v, dtype_ir)
                else:
                    v_scalar = b.global_load(V, v_addr, dtype_ir, align=2)
                v_a_vec = b.vec_insert(v_a_vec, v_scalar, j)
            new_accs[n_blk_atom] = b.mma(
                qk_op,
                p_a_vec,
                v_a_vec,
                new_accs[n_blk_atom],
            )

        # Yield updated state.
        yields = []
        for r in range(atom.c_per_lane):
            yields.append(new_ms[r])
            yields.append(new_ls[r])
        yields.extend(new_accs)
        b.scf_yield(*yields)
        # P_lds will be re-written next iter; sync at top is implied
        # by the next iteration's smem_store path.

    # ---- Pull final state ----
    final = kloop.results
    # ``ms_final`` is the per-lane row-max; the scaled-acc / l_final pair
    # already encodes everything the epilogue needs, so we only consume
    # the normalisation factors below.
    ls_final = [final[2 * r + 1] for r in range(atom.c_per_lane)]
    accs_final = list(final[2 * atom.c_per_lane :])

    # ---- Epilogue: O[m, d] = acc[m, d] / l[m] in target dtype ----
    # Lane t writes to O[q_tile_base + m_blk*4 + r, head, n_blk*16 +
    # n_in_atom] for r in 0..3, n_blk in 0..n_pv_atoms.
    m_blk = b.div(lane, c16)
    for n_blk_atom in range(n_pv_atoms):
        for r in range(atom.c_per_lane):
            o_row = b.add(
                b.add(q_tile_base, b.mul(m_blk, b.const_i32(4))),
                b.const_i32(r),
            )
            o_col = b.add(
                b.mul(b.const_i32(n_blk_atom), b.const_i32(atom.n)),
                m_in_atom,
            )
            # B08: guard ``l_final == 0`` -- when an entire Q tile is
            # masked off (sparse jenga / VSA all-masked rows), the
            # softmax denominator is zero. ``rcp(0) = +inf -> NaN``
            # would poison the output; force ``inv_l = 0`` for the
            # zero case so the contribution evaluates to ``acc * 0
            # == 0`` (the intended "no attention" output).
            inv_l = safe_inv_l(b, ls_final[r])
            v_f32 = b.fmul(b.vec_extract(accs_final[n_blk_atom], r), inv_l)
            if v_scale is not None:
                v_f32 = b.fmul(v_f32, v_scale)
            v_out = b.cast_f32_to(v_f32, dtype_ir)
            addr = b.add(
                b.add(
                    b.mul(o_row, stride_o_token),
                    b.mul(head_idx, stride_o_head),
                ),
                o_col,
            )
            b.global_store(O, addr, v_out, align=2)


# ---------------------------------------------------------------------------
# WMMA (RDNA wave32) FMHA-forward inner body -- the wave32 analogue of the
# MFMA body above.
# ---------------------------------------------------------------------------
#
# This is the *same* QK -> online-softmax -> PV pipeline as
# :func:`mfma_attention_fwd_inner_body`, but every physical fragment fact (which
# lane holds which (row, k) / (k, col) / (row, col) element, how many slots a
# lane owns) is read from the verified gfx1151 ``wmma_f32_16x16x16_f16``
# ``MmaOp`` layout maps, and the matmul is emitted through the target-neutral
# :meth:`IRBuilder.mma`. The wave32 online softmax reuses the arch-parameterized
# :func:`wave_reduce_max` / :func:`wave_reduce_sum` (wave_size=32), which lower
# the in-half XOR butterfly to ``ds_swizzle``.
#
# The fundamental difference from the wave64 MFMA body is the fragment
# distribution: a WMMA accumulator row spans the 16 lanes of one wave32 half and
# each lane owns ``c_frag_len`` (=8) q-rows of one k-column. RDNA3/3.5 (gfx11)
# and RDNA4 (gfx12) differ in the *operand* distribution: on gfx11 the A/B
# fragment carries the full ``a_frag_len`` (=16) K row in every lane (cross-half
# duplication); on gfx12 the duplication is gone -- the fragment is ``<8 x half>``
# per lane and the 16 K-elements of one WMMA step are split across the two lane
# halves (lanes 0-15 carry K 0..7, lanes 16-31 carry K 8..15). The body reads
# *all* of these facts off the per-arch ``MmaOp`` layout maps (``a_frag_len`` and
# the A-operand K coordinate of slot 0 give the per-lane K base), so it never
# hard-codes the wave32 magic numbers and one body serves both RDNA generations.


# Per-arch WMMA attention op_id. gfx11 (RDNA3/3.5) uses the cross-half-duplicated
# ``wmma_f32_16x16x16_*`` atom; gfx12 (RDNA4) uses the split-K
# ``wmma_gfx12_f32_16x16x16_*`` atom (mirrors ``_wmma_params`` in
# ``instances/common/_matmul_nbits_large_n.py``). The op_id also selects the f16
# vs bf16 intrinsic mangling, so it is keyed on the kernel dtype.
def _wmma_attn_op_id(arch: str, dtype: str) -> str:
    elem = "bf16" if dtype == "bf16" else "f16"
    if arch == "gfx1201":
        return f"wmma_gfx12_f32_16x16x16_{elem}"
    return f"wmma_f32_16x16x16_{elem}"


# Default op_id for the historical gfx1151 f16 path (kept for back-references in
# adapters / docs that import the module-level constant).
_WMMA_ATTN_OP_ID = "wmma_f32_16x16x16_f16"


def _wmma_attention_fwd_inner_body(
    b: IRBuilder,
    *,
    Q: Value,
    K: Value,
    V: Value,
    O: Value,  # noqa: E741 - standard attention notation (Q,K,V,O)
    head_size: int,
    seqlen_k: Value,
    q_tile_base: Value,
    head_idx: Value,
    kv_head_idx: Value,
    q_pos_base: Optional[Value],
    stride_q_token: Value,
    stride_q_head: Value,
    stride_k_token: Value,
    stride_k_head: Value,
    stride_v_token: Value,
    stride_v_head: Value,
    stride_o_token: Value,
    stride_o_head: Value,
    scale_log2: Value,
    dtype: str,
    mask_mode: str,
    sliding_window: int,
    causal_ctx_offset: Optional[Value],
    k_token_offset_elems: Optional[Value],
    v_token_offset_elems: Optional[Value],
    k_row_base_fn: Optional[Callable[[IRBuilder, Value], Value]],
    v_row_base_fn: Optional[Callable[[IRBuilder, Value], Value]],
    k_tile_start: Optional[Value],
    k_tile_stop: Optional[Value],
    extra_score_transform: Optional[Callable[[IRBuilder, Value, Value, int], Value]],
    extra_mask_predicate: Optional[Callable[[IRBuilder, Value], Value]],
    extra_skip_predicate: Optional[Callable[[IRBuilder, Value], Value]],
    k_block_iter_fn: Optional[Callable[[IRBuilder, Value], Value]],
    v_scale: Optional[Value],
    v_lds_stage: bool = False,
    arch: str,
    target,
) -> None:
    """One WMMA-tiled QK->softmax->PV pass for a ``BLOCK_M``-row Q tile (wave32).

    Drives the QK^T and PV matmuls through the ``wmma_f32_16x16x16_f16``
    ``MmaOp`` layout maps and :meth:`IRBuilder.mma`; the online softmax uses the
    wave32 row reduction. Parameter semantics match
    :func:`mfma_attention_fwd_inner_body`. The kernel must launch with
    ``block_size == wave_size`` (one wave32 per CTA).
    """
    op_id = _wmma_attn_op_id(arch, dtype)
    op = target.mma.by_op_id(op_id)
    if op is None or op.family != "wmma":
        raise ValueError(f"WMMA attention atom {op_id} absent on {arch}")
    wave = op.wave_size  # 32
    dtype_ir = _ir_type_for_dtype(dtype)

    # Lane/slot coordinate maps come straight from the contract for THIS arch's
    # atom, so the gfx11 (cross-half-duplicated, a_frag=16) and gfx12 (split-K,
    # a_frag=8) ABIs are both expressed through the same accessors.
    a_map = op.a_layout()  # (row, k): lane l -> (row l%16, k=lane-base+slot)
    c_map = (
        op.c_layout()
    )  # (row, col): gfx11 (2i+l//16, l%16); gfx12 ((l//16)*8+i, l%16)
    a_frag = op.a_frag_len  # 16 (gfx11) | 8 (gfx12) -- K elems per lane per step
    c_frag = op.c_frag_len  # 8  -- accumulator slots per lane (same both)

    # Number of WMMA steps along the head-dim axis (QK K-dim == PV N-dim).
    n_dk = head_size // 16

    # Row reduction across the 16 lanes that share one accumulator row. The
    # stage count is derived from the atom geometry (log2(16) = 4), not
    # hard-coded; the XOR masks stay inside the 32-lane half on wave32.
    reduce_stages = wave_reduce_stages(wave_size=wave, lanes_per_row=16)
    assert reduce_stages == 4  # 16x16 tile -> 4-stage butterfly

    lane = b.mod(b.thread_id_x(), b.const_i32(wave))
    c16 = b.const_i32(16)

    # A/B-operand row for this lane (== lane % 16 for both Q and K fragments).
    a_row = a_map.coord(b, lane, 0)[0]
    # gfx12 (RDNA4) split-K: the 16 K-elements of one WMMA step are split across
    # the two lane-halves, so each lane loads ``a_frag`` (=8) elements from K base
    # ``(lane // 16) * a_frag``. gfx11 (RDNA3/3.5) duplicates the full K row in
    # every lane (a_frag=16, base 0). ``split_k`` keeps the gfx11 emission
    # byte-identical (no half-offset add at all) while the gfx12 Q/K/V loads pick
    # up the per-half K offset; mirrors ``split_k_by_half`` in
    # ``instances/common/_matmul_nbits_large_n.py``.
    split_k = a_frag * 2 == 16  # a_frag==8 -> two halves cover K=16 (gfx12)
    k_half_off = b.mul(b.div(lane, c16), b.const_i32(a_frag)) if split_k else None
    # Accumulator column == this lane's k-position in the QK score tile.
    col = b.mod(lane, c16)

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    k_off = k_token_offset_elems if k_token_offset_elems is not None else b.const_i32(0)
    v_off = v_token_offset_elems if v_token_offset_elems is not None else b.const_i32(0)

    # ---- Pre-load Q fragments (constant across the K-loop) ----
    # Lane l holds Q row (q_tile_base + a_row), full <a_frag x half> d-slice for
    # each head-dim WMMA tile. Q is K-loop-invariant -> register-resident.
    q_row = b.add(q_tile_base, a_row)
    q_addr_row_base = b.add(
        b.mul(q_row, stride_q_token),
        b.mul(head_idx, stride_q_head),
    )
    q_frags = []
    for d in range(n_dk):
        q_addr = b.add(q_addr_row_base, b.const_i32(d * 16))
        if k_half_off is not None:
            q_addr = b.add(q_addr, k_half_off)
        q_frags.append(b.global_load_vN(Q, q_addr, dtype_ir, a_frag, align=a_frag * 2))

    # ---- LDS staging tiles ----
    # P_lds transposes the score acc layout -> the PV A-operand layout.
    # V_lds stages the K-tile's V rows once per iteration so the PV B-operand
    # (V in d x k layout) is read from LDS instead of a per-(d,k) scalar
    # global gather.
    P_lds = b.smem_alloc(dtype_ir, [16, 16], name_hint="Pwmma")
    V_lds = (
        b.smem_alloc(dtype_ir, [16, head_size], name_hint="Vwmma")
        if v_lds_stage
        else None
    )

    # ---- Online-softmax + PV accumulator iter-args ----
    iter_args = []
    for r in range(c_frag):
        iter_args.append((f"m{r}", neg_inf))
        iter_args.append((f"l{r}", zero_f))
    for d in range(n_dk):
        iter_args.append((f"acc{d}", b.zero_vec_f32(c_frag)))

    c_block_k = b.const_i32(MFMA_ATTN_BLOCK_K)
    loop_start = k_tile_start if k_tile_start is not None else b.const_i32(0)
    loop_stop = k_tile_stop if k_tile_stop is not None else b.div(seqlen_k, c_block_k)

    kloop = b.scf_for_iter(
        loop_start, loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms = [state[2 * r] for r in range(c_frag)]
        ls = [state[2 * r + 1] for r in range(c_frag)]
        accs = list(state[2 * c_frag :])

        if k_block_iter_fn is not None:
            effective_kt = k_block_iter_fn(b, kt)
        else:
            effective_kt = kt

        k_tile_base = b.mul(effective_kt, c_block_k)
        k_row_for_lane = b.add(k_tile_base, a_row)  # k position for THIS lane

        # Per-K-tile keep / skip predicates (block-sparse). Same semantics as
        # the MFMA body: when false, the score collapses to -inf so the
        # softmax exponential is zero (loads are still issued).
        if extra_mask_predicate is not None:
            keep_tile = extra_mask_predicate(b, kt)
        else:
            keep_tile = None
        if extra_skip_predicate is not None:
            skip_mask = extra_skip_predicate(b, kt)
            keep_tile = (
                b.land(keep_tile, skip_mask) if keep_tile is not None else skip_mask
            )

        if k_row_base_fn is not None:
            k_addr_row_base = k_row_base_fn(b, k_row_for_lane)
        else:
            k_addr_row_base = b.add(
                b.add(
                    b.mul(k_row_for_lane, stride_k_token),
                    b.mul(kv_head_idx, stride_k_head),
                ),
                k_off,
            )

        # ---- QK^T WMMA chain: score = sum_d Q[d-tile] (x) K[d-tile] ----
        score = b.zero_vec_f32(c_frag)
        for d in range(n_dk):
            k_addr = b.add(k_addr_row_base, b.const_i32(d * 16))
            if k_half_off is not None:
                k_addr = b.add(k_addr, k_half_off)
            k_frag = b.global_load_vN(K, k_addr, dtype_ir, a_frag, align=a_frag * 2)
            score = b.mma(op, q_frags[d], k_frag, score)

        # ---- Scale + mask + per-row online softmax ----
        new_ms, new_ls, new_accs = [], [], list(accs)
        ps = []  # per-slot scaled probabilities (acc layout)
        q_pos_for_mask = q_pos_base if q_pos_base is not None else q_tile_base
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)  # (q-row in tile, k-col)
            s_r = b.fmul(b.vec_extract(score, r), scale_log2)
            row_q_pos = b.add(q_pos_for_mask, row_rel)
            k_col_pos = b.add(k_tile_base, col_k)
            if extra_score_transform is not None:
                s_r = extra_score_transform(b, s_r, kt, r)
            s_r = apply_attention_mask(
                b,
                s_r,
                mask_mode=mask_mode,
                k_idx=k_col_pos,
                query_pos=row_q_pos,
                sliding_window=sliding_window,
                context_len=causal_ctx_offset,
            )
            if keep_tile is not None:
                s_r = b.select(keep_tile, s_r, neg_inf)
            # Per-row reduce across the 16 k-columns of this wave32 half. The
            # distribution-driven ``block_tile_reduce_sync`` emits the same
            # 4-stage in-half XOR butterfly as the legacy wave32
            # ``wave_reduce_max(lanes_per_row=16)``.
            row_max = _softmax_row_reduce(b, s_r, combine="max")
            m_new = b.fmax(ms[r], row_max)
            alpha = b.exp2(b.fsub(ms[r], m_new))
            p_r = b.exp2(b.fsub(s_r, m_new))
            row_sum = _softmax_row_reduce(b, p_r, combine="sum")
            l_new = b.fadd(b.fmul(ls[r], alpha), row_sum)
            new_ms.append(m_new)
            new_ls.append(l_new)
            ps.append(p_r)
            for d in range(n_dk):
                old = b.vec_extract(new_accs[d], r)
                new_accs[d] = b.vec_insert(new_accs[d], b.fmul(old, alpha), r)

        # ---- V staging into LDS (vectorized load; transposed PV reads) ----
        # Each lane loads its own k-row's full head_size d-slice as 8-wide
        # vector global loads and writes it row-major into ``V_lds``. The PV
        # B-operand (V in d x k layout) is then a strided *LDS* read, replacing
        # the per-(d,k) scalar global gather the correctness-first version did
        # (it issued ``n_dk * a_frag`` scalar global loads per lane per K-tile).
        # Both wave32 halves map to the same 16 rows (a_row == lane % 16), so
        # the store is redundant across halves but writes identical data.
        if v_lds_stage:
            v_stage_row = b.add(k_tile_base, a_row)
            if v_row_base_fn is not None:
                v_stage_base = v_row_base_fn(b, v_stage_row)
            else:
                v_stage_base = b.add(
                    b.add(
                        b.mul(v_stage_row, stride_v_token),
                        b.mul(kv_head_idx, stride_v_head),
                    ),
                    v_off,
                )
            for e in range(head_size // 8):
                v_g = b.global_load_vN(
                    V, b.add(v_stage_base, b.const_i32(e * 8)), dtype_ir, 8, align=16
                )
                b.smem_store_vN(V_lds, [a_row, b.const_i32(e * 8)], v_g, 8)

        # ---- P staging through LDS: acc layout -> A-operand layout ----
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)
            b.smem_store_vN(P_lds, [row_rel, col_k], b.cast_f32_to(ps[r], dtype_ir), 1)
        b.sync()

        # ---- V load + PV WMMA chain ----
        # PV computes O = P @ V. WMMA evaluates A @ B^T, so B must be V in
        # (d x k) = N x K layout: the B-operand for d-column c is the V *column*
        # c gathered over k = 0..a_frag-1. P A-operand: lane l holds q-row a_row,
        # fragment slot j = P[row, j].
        p_a = b.zero_vec(dtype_ir, a_frag)
        for j in range(a_frag):
            # A-operand layout map gives slot j's (row, k); the row is the
            # loop-invariant ``a_row`` (== lane % 16, hoisted above) so we only
            # take the K coordinate from the map -- the P column to read.
            a_k = a_map.coord(b, lane, j)[1]
            p_v = b.vec_extract(
                b.smem_load_vN(P_lds, a_row, a_k, dtype=dtype_ir, n=1), 0
            )
            p_a = b.vec_insert(p_a, p_v, j)

        for d in range(n_dk):
            d_col = b.add(b.const_i32(d * 16), col)  # this lane's V d-column
            v_b = b.zero_vec(dtype_ir, a_frag)
            for j in range(a_frag):
                # B-operand for d-column ``d_col`` is V[k, d_col]. The K row this
                # lane's slot j feeds is ``j`` on gfx11 (every lane covers the full
                # K, byte-identical to the historical literal) and
                # ``(lane // 16) * a_frag + j`` on gfx12 (split-K halves). The
                # gfx12 base is added via ``k_half_off`` so the gfx11 path emits
                # exactly the previous IR.
                b_k = (
                    b.add(k_half_off, b.const_i32(j))
                    if k_half_off is not None
                    else b.const_i32(j)
                )
                if v_lds_stage:
                    # Optimized: read from the staged LDS tile (V_lds[k, d_col]).
                    v_elem = b.vec_extract(
                        b.smem_load_vN(V_lds, b_k, d_col, dtype=dtype_ir, n=1),
                        0,
                    )
                else:
                    # Baseline: per-(d,k) scalar global gather of V[k, d_col].
                    v_row = b.add(k_tile_base, b_k)
                    if v_row_base_fn is not None:
                        v_row_base = v_row_base_fn(b, v_row)
                    else:
                        v_row_base = b.add(
                            b.add(
                                b.mul(v_row, stride_v_token),
                                b.mul(kv_head_idx, stride_v_head),
                            ),
                            v_off,
                        )
                    v_elem = b.global_load(
                        V, b.add(v_row_base, d_col), dtype_ir, align=2
                    )
                v_b = b.vec_insert(v_b, v_elem, j)
            new_accs[d] = b.mma(op, p_a, v_b, new_accs[d])

        yields = []
        for r in range(c_frag):
            yields.append(new_ms[r])
            yields.append(new_ls[r])
        yields.extend(new_accs)
        b.scf_yield(*yields)

    final = kloop.results
    ls_final = [final[2 * r + 1] for r in range(c_frag)]
    accs_final = list(final[2 * c_frag :])

    # ---- Epilogue: O[q,d] = acc[q,d] / l[q] (zero-denominator guarded) ----
    for d in range(n_dk):
        for r in range(c_frag):
            row_rel, col_n = c_map.coord(b, lane, r)  # (q-row in tile, d-col)
            l_safe = ls_final[r]
            zero_mask = b.fcmp("oeq", l_safe, zero_f)
            inv_l = b.select(zero_mask, zero_f, b.rcp(l_safe))
            v_f32 = b.fmul(b.vec_extract(accs_final[d], r), inv_l)
            if v_scale is not None:
                v_f32 = b.fmul(v_f32, v_scale)
            o_row = b.add(q_tile_base, row_rel)
            o_col = b.add(b.const_i32(d * 16), col_n)
            o_addr = b.add(
                b.add(
                    b.mul(o_row, stride_o_token),
                    b.mul(head_idx, stride_o_head),
                ),
                o_col,
            )
            b.global_store(O, o_addr, b.cast_f32_to(v_f32, dtype_ir), align=2)
