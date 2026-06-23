/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_helpers.mfma_attention.h -- C99 port of selected symbols from
 * ck_dsl/helpers/mfma_attention.py (the MFMA-tiled FMHA-forward inner body, the
 * WMMA wave32 analogue, and the small private helpers they share).
 *
 * SCOPE OF THIS PORT -- exactly these six Python symbols:
 *
 *   Python                                  C99 (this header)
 *   --------------------------------------  -------------------------------------
 *   mfma_attention_fwd_inner_body(...)      ckc_mfma_attention_fwd_inner_body(...)
 *   _softmax_row_reduce(b, scalar, combine) ckc_softmax_row_reduce(...)
 *   _wmma_attention_fwd_inner_body(...)     ckc_wmma_attention_fwd_inner_body(...)
 *   _ir_type_for_dtype(dtype)               ckc_ir_type_for_dtype(...)
 *   _validate_attention_atom(atom, arch)    ckc_validate_attention_atom(...)
 *   _load_kv_dequant_packed(...)            ckc_load_kv_dequant_packed(...)
 *
 * The module-level constants MFMA_ATTN_BLOCK_M / MFMA_ATTN_BLOCK_K are exposed as
 * macros because the body uses them as compile-time tile dims.
 *
 * BINDINGS.
 *   - IR builder primitives: ckc/ir.h's ckc_b_* surface (const_i32/f32, the
 *     arith/float/exp2/rcp/select/land ops, global_load[_vN], global_store,
 *     smem_alloc / smem_store_vN / smem_load_vN, mma, the cvt_* conversions,
 *     vec_extract/vec_insert/vec_concat, vec_cast_f32_to, zero_vec[_f32],
 *     scf_for_iter / scf_yield, sync, thread_id_x, and the type singletons
 *     ckc_f16/ckc_bf16/ckc_f32/ckc_fp8e4m3/ckc_bf8e5m2).
 *   - The MfmaAtom value type is ckc/helper_ck_dsl.helpers.atoms.h's
 *     ckc_mfma_atom_t. The Python factory class-methods
 *     (MfmaAtom.f16_16x16x16 / bf16_16x16x16 / fp8_16x16x32 / bf8_16x16x32 /
 *     fp8_32x32x16 / bf8_32x32x16) and atom.zero_acc() are NOT separate symbols
 *     in atoms.h, so the body resolves each atom via ckc_mfma_atom(dtype,m,n,k)
 *     and inlines zero_acc -> ckc_b_zero_vec_f32(c_per_lane), byte-for-byte.
 *   - Arch / MMA catalog: ckc/arch_target.h (ckc_arch_target_from_gfx,
 *     target->wave_size, target->mma, ckc_mma_catalog_has_shape / by_op_id,
 *     ckc_mma_op_* fragment fields + layout maps, ckc_layout_map_coord).
 *   - Masks / online-softmax helpers: ckc/helper_ck_dsl.helpers.attention.h
 *     (ckc_apply_attention_mask, ckc_safe_inv_l). The mask_mode string is mapped
 *     to ckc_attn_mask_mode_t exactly as that header expects.
 *   - The distribution-driven softmax row reduce binds to
 *     ckc/helper_ck_dsl.helpers.distribution.h (TileDistributionEncoding,
 *     make_static_tile_distribution, make_static_distributed_tensor,
 *     block_tile_reduce_sync) -- the C analogue of _SOFTMAX_ROW_REDUCE_DIST.
 *
 * CALLBACKS. The Python optional Callable closures become explicit C function
 * pointers carrying an opaque `user` cookie. NULL means "the Python None branch".
 * Signatures mirror the Python call shapes one-for-one:
 *   - k_row_base_fn / v_row_base_fn / k_block_iter_fn / extra_mask_predicate /
 *     extra_skip_predicate : (b, value, user) -> value
 *   - extra_score_transform : (b, score_log2, kt, row_in_atom, user) -> value
 *
 * ERROR MODEL. Mirrors the rest of the C port: the sticky-error builder (ckc_b_*)
 * stands in for `raise`. Every Python `raise ValueError` maps onto a
 * ckc_i_set_err(CKC_ERR_VALUE) + early return; the two inner bodies return a
 * ckc_status_t so the caller can observe the raise.
 *
 * FIDELITY. Each function reproduces its Python counterpart's builder-call
 * sequence op-for-op, in the same order, with the same operands and compile-time
 * constants, so the emitted IR is byte-identical. Every emitted node is
 * arena-owned (ckc_ir_builder_t.arena).
 */
#ifndef CKC_HELPER_HELPERS_MFMA_ATTENTION_H
#define CKC_HELPER_HELPERS_MFMA_ATTENTION_H

#include <stdbool.h>

#include "ckc/ir.h"                              /* builder, value, type, status   */
#include "ckc/helper_ck_dsl.helpers.atoms.h"     /* ckc_mfma_atom_t                */
#include "ckc/helper_ck_dsl.helpers.attention.h" /* ckc_attn_mask_mode_t           */

#ifdef __cplusplus
extern "C" {
#endif

/* Module-level tile dims (Python MFMA_ATTN_BLOCK_M / MFMA_ATTN_BLOCK_K). */
#define CKC_MFMA_ATTN_BLOCK_M 16 /* Q rows per CTA per K-tile  */
#define CKC_MFMA_ATTN_BLOCK_K 16 /* K positions per K-tile     */

/* ----------------------------------------------------------- _ir_type_for_dtype *
 *
 * Python:
 *     def _ir_type_for_dtype(dtype):
 *         if dtype in ("f16", "fp16"): return F16
 *         if dtype == "bf16":          return BF16
 *         raise ValueError("... supports f16/bf16; got {dtype!r}")
 *
 * Returns the interned scalar singleton (ckc_f16 / ckc_bf16), or NULL on the
 * Python ValueError path. When `b` is non-NULL the ValueError path also records
 * CKC_ERR_VALUE + the Python message on it. Pass b=NULL for a pure lookup. */
const ckc_type_t* ckc_ir_type_for_dtype(ckc_ir_builder_t* b, const char* dtype);

/* ------------------------------------------------------- _validate_attention_atom *
 *
 * Python:
 *     def _validate_attention_atom(atom, arch) -> None:
 *         cat_dtype = _ATOM_DTYPE_TO_CATALOG.get(atom.dtype_in, atom.dtype_in)
 *         target = ArchTarget.from_gfx(arch)
 *         if not target.mma.has_shape(a_dtype=cat_dtype, b_dtype=cat_dtype,
 *                 c_dtype="fp32", m=atom.m, n=atom.n, k=atom.k):
 *             raise ValueError("mfma_attention: atom ... is not in the {arch}
 *                 MMA catalog; this kernel config is not legal on {arch}")
 *
 * Reject an attention MFMA atom absent from `arch`'s MMA catalog. Returns CKC_OK
 * when the atom IS present (a no-op guard emitting no IR). On the ValueError path
 * (or an unknown gfx / NULL atom) it records CKC_ERR_VALUE + a Python-matching
 * message on the builder and returns that status. */
ckc_status_t
ckc_validate_attention_atom(ckc_ir_builder_t* b, const ckc_mfma_atom_t* atom, const char* arch);

/* ----------------------------------------------------------- _softmax_row_reduce *
 *
 * Python:
 *     def _softmax_row_reduce(b, scalar, *, combine) -> Value:
 *         dt = make_static_distributed_tensor(_SOFTMAX_ROW_REDUCE_DIST, F32)
 *         dt.storage[0] = scalar
 *         block_tile_reduce_sync(b, dt, combine=combine)
 *         return dt.storage[0]
 *
 * Fold the per-lane f32 `scalar` across the 16 lanes sharing one tile row via the
 * distribution-driven block reduce (the 4-stage XOR butterfly, masks 1,2,4,8).
 * `combine` is CKC_REDUCE_MAX (row max) or CKC_REDUCE_SUM (row sum). Returns the
 * reduced per-lane f32, or NULL on a dead builder / construction failure.
 *
 * The single-warp row reduce never runs the cross-warp LDS stage, so the lds_buf
 * / tid arguments to block_tile_reduce_sync are unused; the wave_size arg is
 * passed through but does not affect the emitted lane butterfly. */
ckc_value_t*
ckc_softmax_row_reduce(ckc_ir_builder_t* b, ckc_value_t* scalar, ckc_reduce_combine_t combine);

/* --------------------------------------------------------- _load_kv_dequant_packed *
 *
 * Python:
 *     def _load_kv_dequant_packed(b, *, src, addr, n_elems, kv_dtype_eff,
 *                                 kv_dtype_ir, out_dtype_ir) -> Value:
 *
 * Packed FP8 / BF8 K (or V) load with packed dequant (the P24 hoist). For
 * n_elems % 4 == 0 (and n_elems != 0): one vector global_load_vN, then per
 * 4-element group a packed cvt (cvt_pk_f32_fp8x4 / cvt_pk_f32_bf8x4) assembled
 * from a <4 x kv_dtype_ir> chunk, concat, then vec_cast_f32_to(out_dtype_ir).
 * Otherwise the scalar-dequant fallback: n_elems scalar global_load +
 * cvt_fp8/bf8_to_f32 + cast_f32_to packed via zero_vec + vec_insert.
 *
 * `kv_dtype_eff` is the canonical element name ("fp8e4m3" / "bf8e5m2");
 * `kv_dtype_ir` the matching IR type; `out_dtype_ir` the f16 / bf16 the MFMA
 * atom consumes. Returns the per-lane <n_elems x out_dtype_ir> operand, or NULL
 * on a dead builder. */
ckc_value_t* ckc_load_kv_dequant_packed(ckc_ir_builder_t* b,
                                        ckc_value_t* src,
                                        ckc_value_t* addr,
                                        int n_elems,
                                        const char* kv_dtype_eff,
                                        const ckc_type_t* kv_dtype_ir,
                                        const ckc_type_t* out_dtype_ir);

/* --------------------------------------------------------------------- callbacks *
 *
 * The Python optional Callable closures. NULL == the Python None branch. */

/* (b, value, user) -> value. Covers k_row_base_fn / v_row_base_fn /
 * k_block_iter_fn / extra_mask_predicate / extra_skip_predicate. */
typedef ckc_value_t* (*ckc_attn_value_fn)(ckc_ir_builder_t* b, ckc_value_t* arg, void* user);

/* extra_score_transform: (b, score_log2_per_lane, kt, row_in_atom, user)
 * -> score_log2. */
typedef ckc_value_t* (*ckc_attn_score_xform_fn)(
    ckc_ir_builder_t* b, ckc_value_t* score_log2, ckc_value_t* kt, int row_in_atom, void* user);

/* ---------------------------------------------------- inner-body parameter blocks *
 *
 * The Python helpers take ~40 keyword args; bundling them into a params struct
 * keeps the C call site readable and mirrors the keyword-only Python signature.
 * A field left zero / NULL is the Python default (None / 0 / "" / false). The
 * `user` cookie is forwarded to every callback. */
typedef struct ckc_mfma_attn_fwd_params
{
    /* tensors (global pointers) */
    ckc_value_t* Q;
    ckc_value_t* K;
    ckc_value_t* V;
    ckc_value_t* O;

    int head_size;
    ckc_value_t* seqlen_k;
    ckc_value_t* q_tile_base;
    ckc_value_t* head_idx;
    ckc_value_t* kv_head_idx;
    ckc_value_t* q_pos_base; /* NULL => defaults to q_tile_base                */

    ckc_value_t* stride_q_token;
    ckc_value_t* stride_q_head;
    ckc_value_t* stride_k_token;
    ckc_value_t* stride_k_head;
    ckc_value_t* stride_v_token;
    ckc_value_t* stride_v_head;
    ckc_value_t* stride_o_token;
    ckc_value_t* stride_o_head;

    ckc_value_t* scale_log2;

    const char* dtype;     /* NULL => "f16"                                  */
    const char* mask_mode; /* NULL => "none"                                 */
    int sliding_window;

    ckc_value_t* causal_ctx_offset;    /* NULL => Python None                */
    ckc_value_t* k_token_offset_elems; /* NULL => const_i32(0)               */
    ckc_value_t* v_token_offset_elems; /* NULL => const_i32(0)               */

    ckc_attn_value_fn k_row_base_fn;
    ckc_attn_value_fn v_row_base_fn;
    ckc_value_t* k_tile_start; /* NULL => const_i32(0)                       */
    ckc_value_t* k_tile_stop;  /* NULL => seqlen_k / BLOCK_K                  */

    ckc_attn_score_xform_fn extra_score_transform;
    ckc_attn_value_fn extra_mask_predicate;
    ckc_attn_value_fn extra_skip_predicate;
    ckc_attn_value_fn k_block_iter_fn;

    const char* kv_dtype; /* NULL => same as dtype                           */
    ckc_value_t* v_scale; /* NULL => Python None                             */
    bool use_wider_atom;
    bool native_fp8_path;
    bool use_async_kv;         /* accepted for parity (unused, as in Python)  */
    ckc_value_t* codebook_ptr; /* accepted for parity (unused, as in Python)  */
    bool wmma_v_lds_stage;
    const char* arch; /* NULL => "gfx950"                                */

    void* user; /* forwarded to every callback                              */
} ckc_mfma_attn_fwd_params_t;

/* ----------------------------------------------- mfma_attention_fwd_inner_body *
 *
 * Python:
 *     def mfma_attention_fwd_inner_body(b, *, Q, K, V, O, head_size, seqlen_k,
 *         ..., arch="gfx950") -> None:
 *
 * One MFMA-tiled QK -> online-softmax -> PV pass for a BLOCK_M-row Q tile. On a
 * wave32 target it dispatches to the WMMA body (ckc_wmma_attention_fwd_inner_body)
 * and returns its status. Returns CKC_OK on success; on any Python `raise
 * ValueError` it records CKC_ERR_VALUE on the builder and returns that status.
 * `p` is required (non-NULL). */
ckc_status_t ckc_mfma_attention_fwd_inner_body(ckc_ir_builder_t* b,
                                               const ckc_mfma_attn_fwd_params_t* p);

/* ----------------------------------------------- _wmma_attention_fwd_inner_body *
 *
 * Python:
 *     def _wmma_attention_fwd_inner_body(b, *, ..., v_lds_stage=False, arch,
 *         target) -> None:
 *
 * The wave32 (RDNA / WMMA) analogue of the MFMA body: drives QK^T and PV through
 * the wmma_f32_16x16x16_f16 MmaOp layout maps and the wave32 online-softmax row
 * reduce. Reuses the same params block (fp8 / wider-atom / native-fp8 fields are
 * rejected upstream by the MFMA dispatcher; `v_lds_stage` is taken from
 * p->wmma_v_lds_stage). `target` is the resolved ArchTarget (must be non-NULL --
 * the MFMA dispatcher passes target = ArchTarget.from_gfx(arch)). Returns CKC_OK
 * or the recorded CKC_ERR_VALUE on a raise. */
ckc_status_t ckc_wmma_attention_fwd_inner_body(ckc_ir_builder_t* b,
                                               const ckc_mfma_attn_fwd_params_t* p,
                                               const ckc_arch_target_t* target);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_HELPERS_MFMA_ATTENTION_H */
