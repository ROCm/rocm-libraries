/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_sage_attention_internal.h -- PRIVATE shared state + phase /
 * closure-function contract for the C99 port of _build_sage_mfma and
 * _build_sage_warp (rocke/instances/common/sage_attention.py).
 *
 * WHY THIS HEADER EXISTS.
 *   The two Sage builders are long functions whose bodies share one set of
 *   enclosing-function locals across several nested closures:
 *
 *     _build_sage_mfma (lines 464-591):
 *       captures b, the param Values (Q/K/V/O, q_scale_ptr, k_scale_ptr,
 *       scale_log2_raw, seqlen_k_arg), the decoded grid coords (q_tile_idx,
 *       head_idx, kv_head_idx, q_tile_base, batch_idx), the pre-loaded
 *       q_scale_v, the folded scale_log2, and the per-K-tile K-block-scale
 *       closure (_k_block_scale_transform). That closure reads spec.k_scale,
 *       k_scale_ptr, batch_idx, kv_head_idx and the BLOCK_K constant.
 *
 *     _build_sage_warp (lines 599-812):
 *       captures b, the param Values (Q/K/V/O, q_scale_ptr, k_scale_ptr, cb_k,
 *       cb_v, scale_log2, seqlen_k), the decoded grid coords (q_token, head_idx,
 *       kv_head_idx, batch_idx, tid), the LDS-staged codebook handles, the
 *       pre-loaded q_scale_v, the is_i4 flag, and THREE closures passed as seams
 *       into fmha_warp_fwd_inner_body:
 *         _q_lane_loader      (q_lane_loader seam)
 *         _kv_lane_loader     (kv_lane_loader seam)
 *         _qk_scale_transform (extra_score_transform seam)
 *       all of which read the captured Q/K/V, the codebook handles, the spec
 *       layouts, q_scale_v, batch_idx, the head indices and the dtype/kv_ty.
 *
 *   In C there is no closure capture. The faithful port turns each Python
 *   closure into a free function taking a POINTER to one shared context struct
 *   (rocke_sage_mfma_ctx_t / rocke_sage_warp_ctx_t) holding EXACTLY the variables the
 *   body + closures share. The driver populates the ctx in the same order the
 *   Python prologue computes its locals, then calls the phase functions in Python
 *   order. The closure-seam functions are handed to the helper callbacks via the
 *   helper's `void* user` parameter pointing at the ctx.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing .c TU binds to.
 *   It is DESIGNED TO BE COMPLETE: every local the Python body shares across
 *   phases / closures is a field here. A body agent implementing a phase MUST be
 *   able to read/write only ctx fields and call the prototypes below WITHOUT
 *   editing this header. If a phase genuinely needs a value not present, that is
 *   a design bug to fix here once, deliberately.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `q_tile_base`
 *   -> `ctx->q_tile_base`; Python `q_scale_v` -> `ctx->q_scale_v`). Phase /
 *   closure functions mirror the Python names with a `rocke_sage_mfma_` /
 *   `rocke_sage_warp_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. Included only by the
 * instance_sage_attention*.c translation units. Public callers use
 * rocke/instance_sage_attention.h.
 */
#ifndef ROCKE_INSTANCE_SAGE_ATTENTION_INTERNAL_H
#define ROCKE_INSTANCE_SAGE_ATTENTION_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.mfma_attention.h" /* params + callbacks     */
#include "rocke/helper_rocke.helpers.qk_scale.h" /* rocke_qk_scale_spec_t   */
#include "rocke/helper_rocke.instances.common._fmha_common.h" /* FmhaKernelBuilder      */
#include "rocke/helper_rocke.instances.common._fmha_warp_body.h" /* warp opts + callbacks  */
#include "rocke/instance_sage_attention.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  Codebook table sizes (one f32 entry per quantised level), module-level
 *  constants from sage_attention.py.
 * ===================================================================== */
#define ROCKE_SAGE_CODEBOOK_I8_ENTRIES 256 /* i8 in [-128,127] -> idx = i8 + 128 */
#define ROCKE_SAGE_CODEBOOK_I4_ENTRIES 16 /* i4 in [-8,7]     -> idx = i4 + 8   */

/* Upper bound on `ept` (head_size / WARP_SIZE) for the warp body's per-lane
 * load/dequant scratch. head_size <= 256 in the covered space => ept <= 4; 8 is
 * generous headroom for the per-lane f32 list out-arrays. */
#define ROCKE_SAGE_MAX_EPT 8

/* ===================================================================== *
 *  rocke_sage_mfma_ctx_t  --  shared state for _build_sage_mfma.
 *
 *  Field order follows the Python prologue top-to-bottom (lines 464-591).
 *  This ctx is also the `user` pointer handed to the extra_score_transform
 *  callback (_k_block_scale_transform), so it must carry everything that
 *  closure reads.
 * ===================================================================== */
typedef struct rocke_sage_mfma_ctx
{
    /* ---- inputs / resolved environment -- */
    rocke_fmha_kernel_builder_t* kb; /* the FmhaKernelBuilder `kb`        */
    rocke_ir_builder_t* b; /* kb.builder                        */
    const rocke_sage_attention_spec_t* spec; /* the SageAttentionSpec             */
    const char* arch; /* NULL-normalised "gfx950"          */
    /* spec.common, copied by value for fast field access (s = spec.common). */
    rocke_fmha_common_spec_t s;

    /* ---- kernel params (Values) -- */
    rocke_value_t* Q;
    rocke_value_t* K;
    rocke_value_t* V;
    rocke_value_t* O;
    rocke_value_t* q_scale_ptr; /* kb.ptr("q_scale")                          */
    rocke_value_t* k_scale_ptr; /* kb.ptr("k_scale")                          */
    rocke_value_t* scale_log2_raw; /* kb.scalar("scale_log2")                    */
    rocke_value_t* seqlen_k_arg; /* kb.scalar("seqlen_k")                      */

    /* ---- decoded grid coords (SSA) -- */
    rocke_value_t* q_tile_idx; /* kb.q_token                                 */
    rocke_value_t* head_idx; /* kb.head_idx                                */
    rocke_value_t* kv_head_idx; /* kb.kv_head_idx                             */
    rocke_value_t* q_tile_base; /* q_tile_idx * BLOCK_M                       */
    rocke_value_t* batch_idx; /* const_i32(0)                               */

    /* ---- pre-loaded Q-scale + folded scale_log2 -- */
    rocke_value_t* q_block_idx; /* per_block: magic_div(q_tile_base,scaleblk) *
                               * else const_i32(0)                          */
    rocke_value_t* q_scale_v; /* load_q_scale_for_block(...)                */
    /* For per_head k_scale this is the once-folded K-scale; NULL otherwise. */
    rocke_value_t* k_scale_const;
    rocke_value_t* scale_log2; /* per_head: raw*q*k ; per_block: raw*q       */

    /* ---- per-K-tile K-block-scale closure constants (per_block path only) -- *
     * These are read by rocke_sage_mfma_k_block_scale_transform. */
    rocke_value_t* c_block_k; /* const_i32(BLOCK_K); NULL on per_head path   */

    /* ---- masking / kv-dequant dispatch -- */
    rocke_value_t* causal_ctx; /* const_i32(0) for causal/sliding else NULL   */
    const char* kv_dtype; /* "fp8e4m3" for fp8_bf16 else NULL            */
} rocke_sage_mfma_ctx_t;

/* ===================================================================== *
 *  rocke_sage_warp_ctx_t  --  shared state for _build_sage_warp.
 *
 *  Field order follows the Python prologue top-to-bottom (lines 599-812).
 *  This ctx is the `user` pointer handed to the three warp-body callbacks
 *  (_q_lane_loader, _kv_lane_loader, _qk_scale_transform), so it must carry
 *  everything those closures read.
 * ===================================================================== */
typedef struct rocke_sage_warp_ctx
{
    /* ---- inputs / resolved environment -- */
    rocke_fmha_kernel_builder_t* kb; /* the FmhaKernelBuilder `kb`        */
    rocke_ir_builder_t* b; /* kb.builder                        */
    const rocke_sage_attention_spec_t* spec; /* the SageAttentionSpec             */
    rocke_fmha_common_spec_t s; /* spec.common (by value)            */

    /* ---- derived geometry scalars (Python locals) -- */
    const char* dtype; /* s.dtype                                    */
    int H; /* s.shape.head_size                          */
    int block_size; /* WARP_SIZE                                  */
    const rocke_type_t* kv_ty; /* _kv_pointee_for_quant_mode(quant_mode,dt)  */
    const rocke_type_t* q_pointee; /* io_ir_type(dtype)                          */
    bool is_i4; /* quant_mode == "i4_fp8_bf16"                */
    /* i4-path validated geometry (lines 638-655): one packed byte per lane. */
    int ept_pairs; /* H // (2*WARP_SIZE) (== 1 in v1)            */

    /* ---- kernel params (Values) -- */
    rocke_value_t* Q;
    rocke_value_t* K;
    rocke_value_t* V;
    rocke_value_t* O;
    rocke_value_t* q_scale_ptr; /* kb.ptr("q_scale")                          */
    rocke_value_t* k_scale_ptr; /* kb.ptr("k_scale")                          */
    /* Codebook pointers. After staging (codebook modes) these hold the LDS
     * smem handles returned by _stage_codebook_to_lds; NULL for fp16/fp8. */
    rocke_value_t* cb_k; /* kb.ptr("codebook_k") then LDS handle       */
    rocke_value_t* cb_v; /* kb.ptr("codebook_v") then LDS handle       */
    rocke_value_t* scale_log2; /* kb.scalar("scale_log2")                    */
    rocke_value_t* seqlen_k; /* kb.scalar("seqlen_k")                      */

    /* ---- decoded grid coords + thread id (SSA) -- */
    rocke_value_t* q_token; /* kb.q_token                                 */
    rocke_value_t* head_idx; /* kb.head_idx                                */
    rocke_value_t* kv_head_idx; /* kb.kv_head_idx                             */
    rocke_value_t* batch_idx; /* const_i32(0)                               */
    rocke_value_t* tid; /* thread_id_x()                              */

    /* ---- pre-loaded Q-scale (loaded once for the whole CTA) -- */
    rocke_value_t* q_block_idx; /* per_block: magic_div(q_token,scaleblk)     *
                               * else const_i32(0)                          */
    rocke_value_t* q_scale_v; /* load_q_scale_for_block(...)                */

    /* ---- masking -- */
    rocke_value_t* causal_ctx; /* q_token for causal/sliding else NULL        */
} rocke_sage_warp_ctx_t;

/* ===================================================================== *
 *  SHARED CODEBOOK / LANE-LOAD PRIMITIVES (module-private helpers used by
 *  both the warp body driver and its closures). Ported as free functions
 *  taking the raw builder (they capture no enclosing locals).
 * ===================================================================== */

/* _magic_div(b, dividend, divisor): dividend // divisor via the CK Tile magic
 * mul-hi sequence (calculate_magic_numbers + do_magic_division). `divisor` is a
 * compile-time constant. */
rocke_value_t* rocke_sage_magic_div(rocke_ir_builder_t* b, rocke_value_t* dividend, int divisor);

/* _is_lds_handle(v): true iff `v` is an LDS (smem) allocation handle rather than
 * a global pointer (type(v.type).__name__ == "SmemType"). Dispatches the
 * codebook lookup primitive. */
bool rocke_sage_is_lds_handle(const rocke_value_t* v);

/* _stage_codebook_to_lds(b, cb_global, *, n_entries, tid, name_hint): cooperative
 * one-CTA copy of the constant dequant codebook into LDS. Returns the LDS handle;
 * the caller must b.sync() before reading. Loops when n_entries > WARP_SIZE. */
rocke_value_t* rocke_sage_stage_codebook_to_lds(rocke_ir_builder_t* b,
                                                rocke_value_t* cb_global,
                                                int n_entries,
                                                rocke_value_t* tid,
                                                const char* name_hint);

/* _codebook_lds_lookup_f32(b, cb_lds, idx): one f32 entry from the LDS-staged
 * table (ds_read_b32 -> vec_extract). */
rocke_value_t* rocke_sage_codebook_lds_lookup_f32(rocke_ir_builder_t* b,
                                                  rocke_value_t* cb_lds,
                                                  rocke_value_t* idx);

/* _codebook_i8_to_f32(b, cb_ptr, i8_value): one i8 byte -> f32 via codebook
 * (sext -> +128 -> lookup). cb_ptr is a global ptr OR an LDS handle; the load
 * primitive is chosen from the handle type (rocke_sage_is_lds_handle). */
rocke_value_t* rocke_sage_codebook_i8_to_f32(rocke_ir_builder_t* b,
                                             rocke_value_t* cb_ptr,
                                             rocke_value_t* i8_value);

/* _codebook_i4_pair_to_f32(b, cb_ptr, packed_byte_i8): one packed-i4 byte -> two
 * f32 (unpack_i4_byte_to_pair_i32 -> +8 each -> lookup). Writes the lo / hi
 * Values to *out_lo / *out_hi. */
void rocke_sage_codebook_i4_pair_to_f32(rocke_ir_builder_t* b,
                                        rocke_value_t* cb_ptr,
                                        rocke_value_t* packed_byte_i8,
                                        rocke_value_t** out_lo,
                                        rocke_value_t** out_hi);

/* _vectorised_byte_slice(b, KV, base, lane_d_base, ept, elem_ty): load this
 * lane's `ept` consecutive elements as one vector and write the per-element
 * Values into out[0..ept). ept==1 => scalar load; ept>=2 => global_load_vN with
 * vec_extract (i8 / non-pow2-ept fall back to per-element scalar loads).
 * `out` must have capacity >= ept. */
void rocke_sage_vectorised_byte_slice(rocke_ir_builder_t* b,
                                      rocke_value_t* KV,
                                      rocke_value_t* base,
                                      rocke_value_t* lane_d_base,
                                      int ept,
                                      const rocke_type_t* elem_ty,
                                      rocke_value_t** out);

/* _load_kv_lane_f32(b, *, KV, base, lane_d_base, ept, quant_mode, cb_ptr, kv_ty,
 * dtype): load this lane's full `ept` head-dim K (or V) slice as f32 and write
 * into out[0..ept). fp16/fp8 => vectorised slice + f32 cast; i8 => vectorised
 * byte slice + direct f32 codebook lookup. (i4 is handled by the caller via
 * rocke_sage_codebook_i4_pair_to_f32, NOT this helper.) `out` capacity >= ept.
 * Unsupported quant_mode records ROCKE_ERR_VALUE on the builder. */
void rocke_sage_load_kv_lane_f32(rocke_ir_builder_t* b,
                                 rocke_value_t* KV,
                                 rocke_value_t* base,
                                 rocke_value_t* lane_d_base,
                                 int ept,
                                 rocke_sage_quant_mode_t quant_mode,
                                 rocke_value_t* cb_ptr,
                                 const rocke_type_t* kv_ty,
                                 const char* dtype,
                                 rocke_value_t** out);

/* _load_q_lane_f32(b, Q, q_row_base, lane_d_base, ept, dtype, kv_ty): load this
 * lane's `ept` Q elements as f32 (vectorised when ept in {2,4,8}) into
 * out[0..ept). `out` capacity >= ept. */
void rocke_sage_load_q_lane_f32(rocke_ir_builder_t* b,
                                rocke_value_t* Q,
                                rocke_value_t* q_row_base,
                                rocke_value_t* lane_d_base,
                                int ept,
                                const char* dtype,
                                const rocke_type_t* kv_ty,
                                rocke_value_t** out);

/* ===================================================================== *
 *  ABI DECLARATION (shared between build + signature).
 * ===================================================================== */

/* _declare_params(kb, spec): declare the Sage kernel ABI (Q/K/V/O tensors with
 * the quant-mode-dependent K/V dtype, q_scale/k_scale f32 ptrs, the codebook
 * ptrs for codebook modes, scale_log2/seqlen_q/seqlen_k scalars, and the
 * q/k/v/o strides) into `kb`. Shared by the build paths and the signature
 * probe. */
void rocke_sage_declare_params(rocke_fmha_kernel_builder_t* kb,
                               const rocke_sage_attention_spec_t* spec);

/* _kv_pointee_for_quant_mode(quant_mode, dtype): the K/V tensor pointee IR type
 * (io_ir_type(dtype) for fp16, FP8E4M3 for fp8, I8 otherwise). */
const rocke_type_t* rocke_sage_kv_pointee_for_quant_mode(rocke_sage_quant_mode_t quant_mode,
                                                         const char* dtype);

/* _kv_dtype_str(quant_mode, dtype): ABI dtype string for K/V (dtype for fp16,
 * "fp8e4m3" for fp8, "i8" otherwise). Returns a static / interned string. */
const char* rocke_sage_kv_dtype_str(rocke_sage_quant_mode_t quant_mode, const char* dtype);

/* ===================================================================== *
 *  MFMA-PATH PHASE FUNCTIONS (_build_sage_mfma, lines 464-591).
 *  Each phase reads/writes only ctx (+ the builder it carries) and emits IR in
 *  byte-identical Python order.
 * ===================================================================== */

/* Prologue (lines 477-496): kb = FmhaKernelBuilder(name, common); block_size(64);
 * _declare_params; decode_grid; capture b + all param Values + grid coords;
 * compute q_tile_base and batch_idx. Fills the corresponding ctx fields. */
void rocke_sage_mfma_prologue(rocke_sage_mfma_ctx_t* ctx);

/* Q-scale preload + scale fold (lines 498-555): compute q_block_idx, load
 * q_scale_v; for per_head k_scale fold q*k into scale_log2 (k_scale_const set,
 * c_block_k NULL); for per_block compute scale_log2 = raw*q, set c_block_k =
 * const_i32(BLOCK_K) for the transform closure. Fills q_block_idx / q_scale_v /
 * k_scale_const / scale_log2 / c_block_k. */
void rocke_sage_mfma_fold_scales(rocke_sage_mfma_ctx_t* ctx);

/* extra_score_transform closure _k_block_scale_transform (lines 534-553):
 * (b, score, kt, row_in_atom) -> score * k_scale[k_block_idx(kt)]. Used only on
 * the per_block k_scale path; `user` is the rocke_sage_mfma_ctx_t*. Matches the
 * rocke_attn_score_transform_fn signature so it is passed straight into
 * rocke_mfma_attn_params_t.extra_score_transform. */
rocke_value_t* rocke_sage_mfma_k_block_scale_transform(rocke_ir_builder_t* b,
                                                       rocke_value_t* score_log2,
                                                       rocke_value_t* kt,
                                                       int row_in_atom,
                                                       void* user);

/* Body + epilogue (lines 557-591): resolve causal_ctx / kv_dtype; populate a
 * rocke_mfma_attn_params_t from ctx (wiring extra_score_transform =
 * rocke_sage_mfma_k_block_scale_transform with user=ctx when c_block_k != NULL);
 * call rocke_mfma_attention_fwd_inner_body; b.ret(). Returns the kernel
 * (kb.kernel) on success, NULL on error. */
rocke_kernel_def_t* rocke_sage_mfma_emit_body(rocke_sage_mfma_ctx_t* ctx);

/* ===================================================================== *
 *  WARP-PATH PHASE FUNCTIONS (_build_sage_warp, lines 599-812).
 * ===================================================================== */

/* Prologue (lines 631-677): derive dtype/H/block_size/kv_ty/q_pointee/is_i4;
 * validate the i4 head_size geometry (lines 638-655, raise -> sticky error);
 * kb = FmhaKernelBuilder; block_size(WARP_SIZE); _declare_params; decode_grid;
 * capture b + all param Values (incl cb_k/cb_v ptrs) + grid coords + tid;
 * batch_idx = const_i32(0). Fills the matching ctx fields. Returns false (builder
 * error set) on the i4 geometry violation. */
bool rocke_sage_warp_prologue(rocke_sage_warp_ctx_t* ctx);

/* Codebook staging (lines 679-696): for codebook modes, stage cb_k / cb_v into
 * LDS (rocke_sage_stage_codebook_to_lds, n_entries per mode) and b.sync(); rewrite
 * ctx->cb_k / ctx->cb_v to the LDS handles. No-op for fp16/fp8 (cb_k NULL). */
void rocke_sage_warp_stage_codebooks(rocke_sage_warp_ctx_t* ctx);

/* Q-scale preload (lines 698-712): compute q_block_idx, load q_scale_v. Fills
 * ctx->q_block_idx / ctx->q_scale_v. */
void rocke_sage_warp_preload_q_scale(rocke_sage_warp_ctx_t* ctx);

/* q_lane_loader closure _q_lane_loader (lines 716-718): (b, q_row_base,
 * lane_d_base, ept) -> writes ept f32 Q Values into out_q. Wraps
 * rocke_sage_load_q_lane_f32 with ctx->Q/dtype/q_pointee. `user` == ctx. Matches
 * rocke_fmha_q_lane_loader_fn. */
void rocke_sage_warp_q_lane_loader(rocke_ir_builder_t* b,
                                   rocke_value_t* q_row_base,
                                   rocke_value_t* lane_d_base,
                                   int ept,
                                   rocke_value_t** out_q,
                                   void* user);

/* kv_lane_loader closure _kv_lane_loader (lines 720-756): (b, k_idx, k_row_base,
 * v_row_base, lane_d_base, ept) -> writes ept f32 K Values into out_k and ept f32
 * V Values into out_v. i4: one packed byte per lane -> two nibbles -> two f32 via
 * rocke_sage_codebook_i4_pair_to_f32; else rocke_sage_load_kv_lane_f32 for K and V.
 * `user` == ctx. Matches rocke_fmha_kv_lane_loader_fn. */
void rocke_sage_warp_kv_lane_loader(rocke_ir_builder_t* b,
                                    rocke_value_t* k_idx,
                                    rocke_value_t* k_row_base,
                                    rocke_value_t* v_row_base,
                                    rocke_value_t* lane_d_base,
                                    int ept,
                                    rocke_value_t** out_k,
                                    rocke_value_t** out_v,
                                    void* user);

/* extra_score_transform closure _qk_scale_transform (lines 758-778): (b,
 * score_log2, k_idx) -> apply_qk_scales(score_log2, q_scale_v, k_scale[k_block_idx]).
 * k_scale reloaded per K-iter for per_block. `user` == ctx. Matches
 * rocke_fmha_score_transform_fn. */
rocke_value_t* rocke_sage_warp_qk_scale_transform(rocke_ir_builder_t* b,
                                                  rocke_value_t* score_log2,
                                                  rocke_value_t* k_idx,
                                                  void* user);

/* Body + epilogue (lines 780-812): resolve causal_ctx; populate a
 * rocke_fmha_warp_fwd_opts_t from ctx (wiring q_lane_loader / kv_lane_loader /
 * extra_score_transform to the three closures above with user=ctx); call
 * rocke_fmha_warp_fwd_inner_body; b.ret(). Returns the kernel (kb.kernel) on
 * success, NULL on error. */
rocke_kernel_def_t* rocke_sage_warp_emit_body(rocke_sage_warp_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_SAGE_ATTENTION_INTERNAL_H */
