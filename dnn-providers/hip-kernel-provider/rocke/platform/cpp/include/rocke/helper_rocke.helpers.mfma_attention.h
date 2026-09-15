/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.mfma_attention.h -- C99 port of selected symbols
 * from rocke/helpers/mfma_attention.py (the MFMA-tiled FMHA forward inner-body
 * and its WMMA wave32 analogue).
 *
 * SCOPE OF THIS PORT (this phase) -- exactly these Python symbols:
 *
 *   Python                                  C99 (this header)
 *   --------------------------------------  -------------------------------------
 *   MFMA_ATTN_BLOCK_M                       ROCKE_MFMA_ATTN_BLOCK_M  (macro)
 *   MFMA_ATTN_BLOCK_K                       ROCKE_MFMA_ATTN_BLOCK_K  (macro)
 *   _ir_type_for_dtype(dtype)              rocke_mfma_attn_ir_type_for_dtype(...)
 *   _validate_attention_atom(atom, arch)   rocke_validate_attention_atom(...)
 *   _load_kv_dequant_packed(...)           rocke_load_kv_dequant_packed(...)
 *   _softmax_row_reduce(b, scalar, combine) rocke_softmax_row_reduce(...)
 *   _WMMA_ATTN_OP_ID                       ROCKE_WMMA_ATTN_OP_ID    (macro)
 *   mfma_attention_fwd_inner_body(...)     rocke_mfma_attention_fwd_inner_body(...)
 *   _wmma_attention_fwd_inner_body(...)    rocke_wmma_attention_fwd_inner_body(...)
 *
 * The two private module-level encodings (_SOFTMAX_ROW_REDUCE_ENC /
 * _SOFTMAX_ROW_REDUCE_DIST) are reproduced inside the .c (file scope) because
 * rocke_softmax_row_reduce builds the same one-element StaticDistributedTensor
 * over that reduce distribution.
 *
 * BINDINGS.
 *   - IR builder primitives are rocke/ir.h's rocke_b_* entry points.
 *   - The MfmaAtom value type is rocke/helper_rocke.helpers.atoms.h's
 *     rocke_mfma_atom_t. The atom factory class-methods (f16_16x16x16 etc.) and
 *     atom.zero_acc() that atoms.h does not expose are reproduced inline in the
 *     .c via rocke_mfma_atom(dtype, m, n, k) lookups / rocke_b_zero_vec_f32.
 *   - Arch dispatch (ArchTarget.from_gfx, target.wave_size, target.mma.has_shape,
 *     target.mma.by_op_id, MmaOp.a_layout/c_layout/a_frag_len/c_frag_len) binds
 *     to rocke/helper_rocke.core.arch.h / rocke/arch_target.h.
 *   - The online-softmax row reduce binds to the distribution helper
 *     (rocke/helper_rocke.helpers.distribution.h): the reduce encoding, the
 *     static distribution, the static distributed tensor, block_tile_reduce_sync.
 *   - The mask / inv-l / reduce-stage helpers bind to
 *     rocke/helper_rocke.helpers.attention.h.
 *
 * FIDELITY CONTRACT. Each rocke_* function reproduces its Python counterpart's
 * builder-call sequence op-for-op, in the same order, with the same operands and
 * the same compile-time constants, so the emitted IR is byte-identical to the
 * Python helper's emission.
 *
 * CALLBACKS. The Python closures become explicit C function pointers carrying an
 * opaque `user` pointer. The Python `None` default maps to a NULL function
 * pointer.
 *
 * ERROR MODEL. Mirrors the rest of the C port: the sticky-error builder (rocke_b_*)
 * stands in for `raise`. Each Python `raise ValueError` maps onto a
 * rocke_i_set_err(ROCKE_ERR_VALUE) sticky error.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_MFMA_ATTENTION_H
#define ROCKE_HELPER_ROCKE_HELPERS_MFMA_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arch_target.h" /* rocke_arch_target_t */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t */
#include "rocke/helper_rocke.helpers.attention.h" /* rocke_attn_mask_mode_t */
#include "rocke/helper_rocke.helpers.distribution.h" /* rocke_reduce_combine_t */
#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_value_t, rocke_type_t */

#ifdef __cplusplus
extern "C" {
#endif

/* MFMA_ATTN_BLOCK_M / MFMA_ATTN_BLOCK_K (Python module-level constants). */
#define ROCKE_MFMA_ATTN_BLOCK_M 16 /* Q rows per CTA per K-tile  */
#define ROCKE_MFMA_ATTN_BLOCK_K 16 /* K positions per K-tile     */

/* _WMMA_ATTN_OP_ID (Python module-level constant). */
#define ROCKE_WMMA_ATTN_OP_ID "wmma_f32_16x16x16_f16"

/* ------------------------------------------------------- _ir_type_for_dtype *
 *
 * Python:
 *
 *     def _ir_type_for_dtype(dtype):
 *         if dtype in ("f16", "fp16"): return F16
 *         if dtype == "bf16":          return BF16
 *         raise ValueError("mfma_attention currently supports f16/bf16; got ...")
 *
 * Returns the interned scalar singleton (rocke_f16 / rocke_bf16). On the Python
 * ValueError path it sets the builder sticky error (ROCKE_ERR_VALUE) when `b` is
 * non-NULL and returns NULL. `b` may be NULL for a pure lookup (NULL on miss). */
const rocke_type_t* rocke_mfma_attn_ir_type_for_dtype(rocke_ir_builder_t* b, const char* dtype);

/* --------------------------------------------------- _validate_attention_atom *
 *
 * Python:
 *
 *     def _validate_attention_atom(atom, arch) -> None:
 *         cat_dtype = _ATOM_DTYPE_TO_CATALOG.get(atom.dtype_in, atom.dtype_in)
 *         target = ArchTarget.from_gfx(arch)
 *         if not target.mma.has_shape(a_dtype=cat_dtype, b_dtype=cat_dtype,
 *                 c_dtype="fp32", m=atom.m, n=atom.n, k=atom.k):
 *             raise ValueError("mfma_attention: atom ... is not in the {arch}
 *                 MMA catalog; this kernel config is not legal on {arch}")
 *
 * Guard the QK/PV atom against the per-arch MMA catalog. Returns ROCKE_OK when the
 * atom IS in the catalog. On the ValueError path it records ROCKE_ERR_VALUE + a
 * Python-matching message on the builder and returns that status. An unknown
 * `arch` resolves the target to NULL, reported as a ROCKE_ERR_VALUE no-target
 * error. `atom` must be non-NULL. */
rocke_status_t rocke_validate_attention_atom(rocke_ir_builder_t* b,
                                             const rocke_mfma_atom_t* atom,
                                             const char* arch);

/* --------------------------------------------------- _load_kv_dequant_packed *
 *
 * Python:
 *
 *     def _load_kv_dequant_packed(b, *, src, addr, n_elems, kv_dtype_eff,
 *                                 kv_dtype_ir, out_dtype_ir) -> Value:
 *
 * Packed FP8 / BF8 K (or V) load with packed dequant. For n_elems % 4 != 0 (or
 * 0) it falls back to n_elems scalar global_load + cvt_fp8/bf8_to_f32 +
 * cast_f32_to. Otherwise one global_load_vN(n_elems) + grouped cvt_pk_f32_*x4 +
 * vec_concat + vec_cast_f32_to. `kv_dtype_eff` is "fp8e4m3" or "bf8e5m2".
 * Returns the per-lane <n_elems x out_dtype_ir> operand, or NULL on a dead
 * builder. */
rocke_value_t* rocke_load_kv_dequant_packed(rocke_ir_builder_t* b,
                                            rocke_value_t* src,
                                            rocke_value_t* addr,
                                            int n_elems,
                                            const char* kv_dtype_eff,
                                            const rocke_type_t* kv_dtype_ir,
                                            const rocke_type_t* out_dtype_ir);

/* ------------------------------------------------------- _softmax_row_reduce *
 *
 * Python:
 *
 *     def _softmax_row_reduce(b, scalar, *, combine) -> Value:
 *         dt = make_static_distributed_tensor(_SOFTMAX_ROW_REDUCE_DIST, F32)
 *         dt.storage[0] = scalar
 *         block_tile_reduce_sync(b, dt, combine=combine)
 *         return dt.storage[0]
 *
 * Reduce `scalar` across the 16 lanes sharing one tile row (CK Tile
 * BlockReduce2dSync over the module-level reduce distribution). `combine` is
 * ROCKE_REDUCE_MAX or ROCKE_REDUCE_SUM. Returns the folded per-lane f32, or NULL on
 * a dead builder / distribution failure. */
rocke_value_t* rocke_softmax_row_reduce(rocke_ir_builder_t* b,
                                        rocke_value_t* scalar,
                                        rocke_reduce_combine_t combine);

/* --------------------------------------------------------------- callbacks *
 *
 * The Python keyword callbacks become C function pointers + an opaque `user`.
 * NULL pointer == Python `None` (the `is not None` guard).
 *
 *   k_row_base_fn / v_row_base_fn : Callable[[IRBuilder, Value], Value]
 *       (b, row_idx) -> i32 element offset for one K/V row.
 *   extra_score_transform         : Callable[[IRBuilder, Value, Value, int], Value]
 *       (b, score_log2, kt, row_in_atom) -> score_log2.
 *   extra_mask_predicate          : Callable[[IRBuilder, Value], Value]
 *       (b, kt) -> i1 per-K-tile keep flag.
 *   extra_skip_predicate          : Callable[[IRBuilder, Value], Value]
 *       (b, kt) -> i1 per-K-tile skip flag.
 *   k_block_iter_fn               : Callable[[IRBuilder, Value], Value]
 *       (b, kt) -> effective K-tile index (block-sparse LUT). */
typedef rocke_value_t* (*rocke_attn_row_base_fn)(rocke_ir_builder_t* b,
                                                 rocke_value_t* row_idx,
                                                 void* user);
typedef rocke_value_t* (*rocke_attn_score_transform_fn)(rocke_ir_builder_t* b,
                                                        rocke_value_t* score_log2,
                                                        rocke_value_t* kt,
                                                        int row_in_atom,
                                                        void* user);
typedef rocke_value_t* (*rocke_attn_predicate_fn)(rocke_ir_builder_t* b,
                                                  rocke_value_t* kt,
                                                  void* user);
typedef rocke_value_t* (*rocke_attn_kblock_iter_fn)(rocke_ir_builder_t* b,
                                                    rocke_value_t* kt,
                                                    void* user);

/* ------------------------------------------------------------- parameters *
 *
 * The Python helper has a long keyword-only argument list; the C port bundles it
 * into one params struct (clearer than a 40-arg function and stable across the
 * MFMA / WMMA dispatch). Optional Value args map to NULL (Python `None`); the
 * callbacks map to NULL function pointers. Field order mirrors the Python
 * signature. `mask_mode` is the enum form of the Python "none"/"causal"/
 * "sliding_window" string. */
typedef struct rocke_mfma_attn_params
{
    rocke_value_t* Q;
    rocke_value_t* K;
    rocke_value_t* V;
    rocke_value_t* O;
    int head_size;
    rocke_value_t* seqlen_k;
    rocke_value_t* q_tile_base;
    rocke_value_t* head_idx;
    rocke_value_t* kv_head_idx;
    rocke_value_t* q_pos_base; /* NULL => default q_tile_base               */

    rocke_value_t* stride_q_token;
    rocke_value_t* stride_q_head;
    rocke_value_t* stride_k_token;
    rocke_value_t* stride_k_head;
    rocke_value_t* stride_v_token;
    rocke_value_t* stride_v_head;
    rocke_value_t* stride_o_token;
    rocke_value_t* stride_o_head;

    rocke_value_t* scale_log2;
    const char* dtype; /* NULL => "f16"                       */
    rocke_attn_mask_mode_t mask_mode; /* ROCKE_ATTN_MASK_NONE default          */
    int sliding_window;
    rocke_value_t* causal_ctx_offset; /* NULL => None                    */
    rocke_value_t* k_token_offset_elems; /* NULL => const_i32(0)            */
    rocke_value_t* v_token_offset_elems; /* NULL => const_i32(0)            */

    rocke_attn_row_base_fn k_row_base_fn;
    void* k_row_base_user;
    rocke_attn_row_base_fn v_row_base_fn;
    void* v_row_base_user;

    rocke_value_t* k_tile_start; /* NULL => const_i32(0)                     */
    rocke_value_t* k_tile_stop; /* NULL => seqlen_k / BLOCK_K               */

    rocke_attn_score_transform_fn extra_score_transform;
    void* extra_score_transform_user;
    rocke_attn_predicate_fn extra_mask_predicate;
    void* extra_mask_predicate_user;
    rocke_attn_predicate_fn extra_skip_predicate;
    void* extra_skip_predicate_user;
    rocke_attn_kblock_iter_fn k_block_iter_fn;
    void* k_block_iter_user;

    const char* kv_dtype; /* NULL / "f16" / "fp8e4m3" / "bf8e5m2"          */
    rocke_value_t* v_scale; /* NULL => None                                  */
    bool use_wider_atom;
    bool native_fp8_path;
    bool use_async_kv;
    rocke_value_t* codebook_ptr;
    bool wmma_v_lds_stage;
    const char* arch; /* NULL => "gfx950"                                  */
} rocke_mfma_attn_params_t;

/* ---------------------------------------------- mfma_attention_fwd_inner_body *
 *
 * Python:
 *
 *     def mfma_attention_fwd_inner_body(b, *, Q, K, V, O, head_size, ...) -> None
 *
 * One MFMA-tiled QK->softmax->PV pass for a BLOCK_M-row Q tile. On a wave32
 * (RDNA) target it dispatches to rocke_wmma_attention_fwd_inner_body; on a wave64
 * (CDNA) target it emits the MFMA body. Returns ROCKE_OK, or the builder sticky
 * status on any of the Python raise paths (bad head_size / dtype / kv_dtype, or
 * an atom not in the arch catalog). `p` must be non-NULL. */
rocke_status_t rocke_mfma_attention_fwd_inner_body(rocke_ir_builder_t* b,
                                                   const rocke_mfma_attn_params_t* p);

/* ---------------------------------------------- _wmma_attention_fwd_inner_body *
 *
 * Python:
 *
 *     def _wmma_attention_fwd_inner_body(b, *, ..., v_lds_stage=False, arch,
 *                                        target) -> None
 *
 * The wave32 (WMMA) analogue of the MFMA body, driven off the
 * wmma_f32_16x16x16_f16 MmaOp layout maps. Normally reached via the dispatch in
 * rocke_mfma_attention_fwd_inner_body, but exposed directly for parity with the
 * Python symbol. `target` is the resolved ArchTarget (ArchTarget.from_gfx(arch));
 * pass NULL to have the function resolve it from `p->arch`. `v_lds_stage` mirrors
 * the Python keyword (the MFMA dispatch passes p->wmma_v_lds_stage). Returns
 * ROCKE_OK or the builder sticky status. */
rocke_status_t rocke_wmma_attention_fwd_inner_body(rocke_ir_builder_t* b,
                                                   const rocke_mfma_attn_params_t* p,
                                                   bool v_lds_stage,
                                                   const rocke_arch_target_t* target);

/* ===========================================================================
 * Additional symbols ported from rocke.helpers.attention (the real home of
 * mfma_32x32x16_for_dtype / dequant_fp8x8_to_dtype) into this module's surface,
 * per the C-port symbol request. These are byte-faithful reproductions of the
 * Python builder-call sequences. The module-prefixed names
 * (rocke_mfma_attn_*) keep external linkage distinct from any other module that
 * also ports the same Python function.
 * ===========================================================================
 */

/* ------------------------------------------------------- mfma_32x32x16_for_dtype *
 *
 * Python (rocke/helpers/attention.py):
 *
 *     def mfma_32x32x16_for_dtype(b, dtype, a, bv, c) -> Value:
 *         if dtype.name == "f16":     return b.mfma_f32_32x32x16_f16(a, bv, c)
 *         if dtype.name == "bf16":    return b.mfma_f32_32x32x16_bf16(a, bv, c)
 *         if dtype.name == "fp8e4m3": return b.mfma_f32_32x32x16_fp8(a, bv, c)
 *         raise ValueError(f"unsupported MFMA 32x32x16 dtype {dtype.name}")
 *
 * Dispatches the 32x32x16 MFMA atom (per-lane A/B: <8 x dtype>, C/D: <16 x f32>)
 * for f16 / bf16 / fp8e4m3. Returns the per-lane <16 x f32> accumulator, or NULL
 * (sticky ROCKE_ERR_VALUE) for a NULL/unsupported dtype. */
rocke_value_t* rocke_mfma_attn_mfma_32x32x16_for_dtype(rocke_ir_builder_t* b,
                                                       const rocke_type_t* dtype,
                                                       rocke_value_t* a,
                                                       rocke_value_t* bv,
                                                       rocke_value_t* c);

/* ------------------------------------------------------- dequant_fp8x8_to_dtype *
 *
 * Python (rocke/helpers/attention.py):
 *
 *     def dequant_fp8x8_to_dtype(b, fp8_vec, scale, dtype) -> Value:
 *         lo_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4)], FP8E4M3)
 *         hi_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4, 8)], FP8E4M3)
 *         lo_f32 = b.cvt_pk_f32_fp8x4(lo_fp8)
 *         hi_f32 = b.cvt_pk_f32_fp8x4(hi_fp8)
 *         deq = [b.cast_f32_to(b.fmul(b.vec_extract(lo_f32, i), scale), dtype) for i in range(4)]
 *             + [b.cast_f32_to(b.fmul(b.vec_extract(hi_f32, i), scale), dtype) for i in range(4)]
 *         return b.vec_pack(deq, dtype)
 *
 * In-register dequant of <8 x fp8e4m3> to a packed <8 x dtype>. The scale is an
 * UNFUSED explicit fmul (NOT the fused E8M0-scale cvt) so arbitrary non-pow2
 * scales stay exact. Returns NULL (sticky ROCKE_ERR_VALUE) if `dtype` is NULL. */
rocke_value_t* rocke_mfma_attn_dequant_fp8x8_to_dtype(rocke_ir_builder_t* b,
                                                      rocke_value_t* fp8_vec,
                                                      rocke_value_t* scale,
                                                      const rocke_type_t* dtype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_MFMA_ATTENTION_H */
