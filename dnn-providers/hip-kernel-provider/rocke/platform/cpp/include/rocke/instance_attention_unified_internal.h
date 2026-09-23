/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_unified_internal.h -- PRIVATE shared state + phase-
 * function contract for the C99 port of the SCALAR unified-attention kernel
 * builders (rocke/instances/common/attention_unified.py:
 *   build_unified_attention_2d        lines 2589-2774
 *   build_unified_attention_3d        lines 2798-2914
 *   build_unified_attention_reduce    lines 2941-3009
 * plus the module helpers they thread:
 *   _declare_scalar_attn_params       3037-3097
 *   _emit_find_seq_idx_scan           3012-3034
 *   _q_descriptor                     3100-3106
 *   _paged_kv_descriptor              3109-3117
 *   _segm_descriptors                 3120-3151
 *   _emit_qk_score                    3154-3231
 *   _emit_v_load                      3234-3255
 *   _physical_block_and_token         3258-3270
 *   _magic_div / _magic_div_mod       175-201
 *   _apply_softcap (= apply_softcap_log2)).
 *
 * WHY THIS HEADER EXISTS.
 *   Each build_unified_attention_* function in Python is a flat-but-long body
 *   that (a) declares the shared scalar ABI prefix via _declare_scalar_attn_params
 *   and pulls each Value out of the returned dict, (b) computes a prologue stack
 *   of grid ids / seq-idx / cu_q bounds / kv geometry / SSA constants, (c)
 *   threads the descriptor + emit helpers (_emit_qk_score / _emit_v_load /
 *   _physical_block_and_token / _segm_descriptors / _q_descriptor) inside one or
 *   more scf.for online-softmax loops, and (d) emits a guarded epilogue store.
 *   The three builders SHARE the same ABI prefix declaration, the same seq-idx
 *   scan, the same magic-div geometry, and the same QK/V emit helpers.
 *
 *   In C there is no closure capture and no Python dict-of-Values. The faithful
 *   port turns each Python helper into a free function that takes a POINTER to
 *   one shared context struct, rocke_attn_unified_build_ctx_t, which holds EXACTLY
 *   the set of Values + geometry + descriptors the builders and helpers share.
 *   Each public build entry (in the glue bucket) zero-inits a ctx, populates the
 *   problem/spec slice + ABI prefix + prologue in the SAME order as the Python
 *   prologue, then calls the phase functions in Python execution order.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing .c binds to.
 *   It is DESIGNED TO BE COMPLETE: every local/shared variable the Python bodies
 *   and helpers pass around is a field here. A body agent implementing a phase
 *   .c file MUST be able to read/write only ctx fields and call the prototypes
 *   below WITHOUT editing this header. If a phase genuinely needs a value not
 *   present, that is a design bug in this header to be fixed once, deliberately,
 *   not patched per-phase.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `q_tok` ->
 *   `ctx->q_tok`; Python `context_len` -> `ctx->context_len`; the
 *   _declare_scalar_attn_params dict keys -> the abi_* fields). Phase functions
 *   mirror the Python module-helper / build-function names with a
 *   `rocke_attn_unified_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_attention_unified_*.c translation units. Public callers use
 * rocke/instance_attention_unified.h.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_UNIFIED_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_UNIFIED_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.transforms.h" /* rocke_tensor_descriptor_t */
#include "rocke/instance_attention_unified.h"
#include "rocke/ir.h"
/* Re-uses the already-ported selector/descriptor/emit surface (paged-KV
 * descriptor type, magic-div, _emit_qk_score / _emit_v_load /
 * _physical_block_and_token, _q_descriptor / _segm_descriptors). */
#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * Which kernel a ctx is being populated for (selects ABI tail +
 * epilogue / loop body, all sharing the same prologue + ABI prefix).
 * ============================================================ */
typedef enum rocke_attn_unified_kind
{
    ROCKE_ATTN_UNIFIED_2D = 0, /* build_unified_attention_2d     */
    ROCKE_ATTN_UNIFIED_3D, /* build_unified_attention_3d     */
    ROCKE_ATTN_UNIFIED_REDUCE /* build_unified_attention_reduce */
} rocke_attn_unified_kind_t;

/* ===================================================================== *
 *  rocke_attn_unified_build_ctx_t
 *
 *  The single shared state object. Holds every variable the Python build
 *  bodies + module helpers pass around, grouped by the Python prologue's
 *  phases. Populated by the glue driver; consumed by the phase functions.
 * ===================================================================== */
typedef struct rocke_attn_unified_build_ctx
{
    /* ---------- inputs / configuration (Python `spec`, `p`, `dtype`) ---------- */
    rocke_ir_builder_t* b; /* the IRBuilder (Python `b`)   */
    rocke_attn_unified_kind_t kind; /* which build_* is running      */
    const rocke_unified_attention_problem_t* p; /* Python `p = spec.problem`     */
    /* Mirror of `p` in the selector-helper struct shape, so the ported
     * emit / physical-block-and-token / descriptor helpers (which take
     * rocke_unified_attn_problem_t*) can be called without re-deriving it. The
     * glue driver fills this from `p` once. */
    rocke_unified_attn_problem_t sel_p;
    const rocke_type_t* dtype; /* spec.dtype_ir (F16 / BF16)    */
    int num_segments; /* 3D/reduce; 0 for 2D           */
    int num_queries_per_kv; /* p->num_query_heads / kv       */
    int max_blocks; /* ceil(max_seqlen_k/block_size) */

    /* ---------- kernel result + extra outputs ---------- */
    rocke_kernel_def_t* kernel; /* == b->kernel; returned by the driver          */
    /* 2D: single `output` param. 3D: segm_output / segm_max / segm_expsum params.
     * reduce: `out` + segm_output / segm_max / segm_expsum read params. */
    rocke_value_t* output; /* 2D output_ptr / reduce output_ptr (dtype*)    */
    rocke_value_t* segm_output; /* 3D writeonly / reduce readonly (F32*)         */
    rocke_value_t* segm_max; /* 3D writeonly / reduce readonly (F32*)         */
    rocke_value_t* segm_expsum; /* 3D writeonly / reduce readonly (F32*)         */

    /* ---------- shared scalar ABI prefix (_declare_scalar_attn_params dict) ----
     * Declared in EXACTLY the Python order (query/key/value, then sink,
     * block_tables, seq_lens, alibi, qq_bias, cu_q, then scale/k_scale/v_scale).
     * Param declaration order is load-bearing (fixes the kernel arg ABI). The
     * reduce kernel declares only seq_lens + cu_q of this set (its own subset),
     * so the unused fields stay NULL there. */
    rocke_value_t* abi_query; /* "query_ptr"            (dtype*)              */
    rocke_value_t* abi_key; /* "key_cache_ptr"        (dtype*)              */
    rocke_value_t* abi_value; /* "value_cache_ptr"      (dtype*)              */
    rocke_value_t* abi_sink; /* "sink_ptr"             (dtype*)              */
    rocke_value_t* abi_block_tables; /* "block_tables_ptr"     (I32*)               */
    rocke_value_t* abi_seq_lens; /* "seq_lens_ptr"         (I32*)               */
    rocke_value_t* abi_alibi; /* "alibi_slopes_ptr"     (F32*)               */
    rocke_value_t* abi_qq_bias; /* "qq_bias_ptr"          (F32*)               */
    rocke_value_t* abi_cu_q; /* "query_start_len_ptr"  (I32*)               */
    rocke_value_t* abi_scale; /* "scale"                (F32)                */
    rocke_value_t* abi_k_scale; /* "k_scale"              (F32)                */
    rocke_value_t* abi_v_scale; /* "v_scale"              (F32)                */

    /* ---------- kernel-specific tail params (after the ABI prefix) ----------
     * 2D appends out_scale, softcap, num_seqs. 3D appends softcap (unused) +
     * num_seqs. reduce appends none. */
    rocke_value_t* out_scale; /* 2D "out_scale" (F32; unused by body)        */
    rocke_value_t* softcap; /* 2D/3D "softcap" (F32; 3D unused)            */
    rocke_value_t* num_seqs; /* 2D/3D "num_seqs" (I32)                       */

    /* ---------- grid ids + thread predicate (prologue) ---------- */
    rocke_value_t* q_tok; /* block_id_x()                                 */
    rocke_value_t* q_head; /* block_id_y()                                 */
    rocke_value_t* dim; /* 2D/reduce: block_id_z(); 3D: low of zd       */
    rocke_value_t* zd; /* 3D: block_id_z() (before magic_div_mod)      */
    rocke_value_t* segm_idx; /* 3D: high of magic_div_mod(zd, head_size)     */
    rocke_value_t* tid; /* thread_id_x()                                */
    rocke_value_t* active; /* cmp_eq(tid, 0)                               */

    /* ---------- per-sequence geometry (prologue) ---------- */
    rocke_value_t* seq_idx; /* seq-idx scan result                          */
    rocke_value_t* cu_start; /* cu_q[seq_idx]                                */
    rocke_value_t* cu_stop; /* cu_q[seq_idx+1]                              */
    rocke_value_t* q_len; /* cu_stop - cu_start                           */
    rocke_value_t* query_pos; /* q_tok - cu_start                             */
    rocke_value_t* kv_len; /* seq_lens[seq_idx]                            */
    rocke_value_t* context_len; /* kv_len - q_len                               */
    rocke_value_t* kv_head; /* magic_div(q_head, num_queries_per_kv)        */

    /* ---------- 3D segment span (prologue) ---------- */
    rocke_value_t* tiles_per_segment; /* ceil(kv_len/(num_segments*block_size))      */
    rocke_value_t* seg_start; /* segm_idx * tiles_per_segment * block_size    */
    rocke_value_t* seg_stop_i; /* min((segm_idx+1)*..., kv_len)                */

    /* ---------- SSA constants (prologue) ---------- */
    rocke_value_t* neg_inf; /* const_f32(-inf)                              */
    rocke_value_t* zero_f; /* const_f32(0.0)                               */
    rocke_value_t* one_f; /* const_f32(1.0)                               */
    rocke_value_t* rcp_ln2; /* const_f32(1.4426950408889634)                */

    /* ---------- online-softmax loop init (kind-dependent) ----------
     * 2D: init_m = sinks? sink_h*rcp_ln2 : -inf; init_l = 1.0; init_acc = 0.
     * 3D: init_m = -inf; init_l = 0.0; init_acc = 0. */
    rocke_value_t* init_m;
    rocke_value_t* init_l;
    rocke_value_t* init_acc;
    rocke_value_t* sink_h; /* 2D + use_sinks: global_load(sinks, q_head)   */

    /* ---------- descriptors (shared by all three kernels) ----------
     * q_desc      : Q/output naive descriptor (token, head, dim).
     * kv_desc     : element-unit paged-KV descriptor (value type).
     * ml_desc     : 3D/reduce segm_ml descriptor (token, head, seg).
     * out_desc    : 3D/reduce segm_output descriptor (token, head, seg, dim). */
    rocke_tensor_descriptor_t* q_desc;
    rocke_unified_attn_paged_kv_descriptor_t kv_desc; /* value type, not a pointer  */
    rocke_tensor_descriptor_t* ml_desc;
    rocke_tensor_descriptor_t* out_desc;

    /* ---------- online-softmax loop results (epilogue inputs) ----------
     * After the scf.for the driver stashes (m, l, acc) here for the epilogue.
     * 2D/3D: loop_m/loop_l/loop_acc. reduce: overall (max), then den/acc. */
    rocke_value_t* loop_m; /* loop.results[0]                              */
    rocke_value_t* loop_l; /* loop.results[1]                              */
    rocke_value_t* loop_acc; /* loop.results[2]                              */
    rocke_value_t* overall; /* reduce: max-loop result                      */
    rocke_value_t* red_den; /* reduce: red.results[0]                       */
    rocke_value_t* red_acc; /* reduce: red.results[1]                       */
} rocke_attn_unified_build_ctx_t;

/* ============================================================ *
 * Shared prologue + ABI phase functions (used by all 3 builders)
 * ============================================================ */

/* Zero-init the ctx and copy in the problem/spec slice (b, kind, p, sel_p,
 * dtype, num_segments) + derive num_queries_per_kv / max_blocks. On a non-
 * divisible head ratio or unsupported dtype, sets b's sticky error and returns
 * false. */
bool rocke_attn_unified_ctx_init(rocke_attn_unified_build_ctx_t* ctx,
                                 rocke_ir_builder_t* b,
                                 rocke_attn_unified_kind_t kind,
                                 const rocke_unified_attention_problem_t* p,
                                 const rocke_type_t* dtype,
                                 int num_segments);

/* Python: _declare_scalar_attn_params(b, dtype). Declares the shared ABI prefix
 * in the load-bearing Python order, filling ctx->abi_*. (Reduce uses its own
 * narrower declaration -- see rocke_attn_unified_declare_reduce_params.) */
void rocke_attn_unified_declare_scalar_params(rocke_attn_unified_build_ctx_t* ctx);

/* Python: _emit_find_seq_idx_scan(b, cu_q, q_tok, num_seqs) -> seq_idx (3D), and
 * the inline linear cu_q scan (2D). Emits the scan and fills ctx->seq_idx.
 * (2D uses the inline linear scf.for scan; 3D delegates to the bounded binary
 * search -- numerically identical seq_idx.) */
void rocke_attn_unified_emit_find_seq_idx(rocke_attn_unified_build_ctx_t* ctx);

/* Common prologue after the ABI prefix + seq-idx scan: grid ids, the cu_q
 * bounds / q_len / query_pos / kv_len / context_len / kv_head geometry, and the
 * SSA constants (neg_inf / zero_f / one_f / rcp_ln2). Fills the corresponding
 * ctx fields. (For 3D it also computes segm_idx/dim from zd and the segment
 * span seg_start/seg_stop_i/tiles_per_segment.) */
void rocke_attn_unified_emit_prologue(rocke_attn_unified_build_ctx_t* ctx);

/* ============================================================ *
 * 2D scalar kernel phases (build_unified_attention_2d)
 * ============================================================ */

/* The 2D online-softmax over [0, kv_len): per-iter physical/token resolve,
 * inline vec8 QK dot + scalar tail, scale*rcp_ln2 (+ softcap), causal/sliding
 * mask, online (m,l,acc) update with the -inf-masked-row guard, and the single
 * V element fold. Reads ctx prologue/desc fields; writes ctx->loop_m/l/acc. */
void rocke_attn_unified_emit_2d_softmax_loop(rocke_attn_unified_build_ctx_t* ctx);

/* The 2D guarded epilogue: out_val = acc * rcp(l); cast to dtype; store at
 * q_desc.offset(q_tok,q_head,dim) under (active && dim < head_size). */
void rocke_attn_unified_emit_2d_epilogue(rocke_attn_unified_build_ctx_t* ctx);

/* ============================================================ *
 * 3D scalar segment kernel phases (build_unified_attention_3d)
 * ============================================================ */

/* The 3D segment online-softmax over [seg_start, seg_stop_i): uses the shared
 * _emit_qk_score + _emit_v_load helpers, causal mask only, online (m,l,acc)
 * update. Writes ctx->loop_m/l/acc. */
void rocke_attn_unified_emit_3d_segment_loop(rocke_attn_unified_build_ctx_t* ctx);

/* The 3D guarded epilogue: store acc into segm_output at
 * out_desc.offset(token,head,seg,dim); under (active && dim==0) store m,l into
 * segm_max/segm_expsum at ml_desc.offset(token,head,seg). */
void rocke_attn_unified_emit_3d_epilogue(rocke_attn_unified_build_ctx_t* ctx);

/* ============================================================ *
 * Reduce scalar kernel phases (build_unified_attention_reduce)
 * ============================================================ */

/* Reduce-specific narrow ABI: output_ptr + segm_output/segm_max/segm_expsum +
 * seq_lens + cu_q (only this subset of the scalar prefix). Fills the relevant
 * ctx->output / segm_* / abi_seq_lens / abi_cu_q + grid ids. */
void rocke_attn_unified_declare_reduce_params(rocke_attn_unified_build_ctx_t* ctx);

/* The reduce max-pass over [0, num_segments): overall = max(segm_max[..,seg]).
 * Writes ctx->overall. */
void rocke_attn_unified_emit_reduce_max_loop(rocke_attn_unified_build_ctx_t* ctx);

/* The reduce combine-pass over [0, num_segments): den += l*exp2(m-overall);
 * acc += segm_output[..,seg,dim]*exp2(m-overall). Writes ctx->red_den/red_acc. */
void rocke_attn_unified_emit_reduce_combine_loop(rocke_attn_unified_build_ctx_t* ctx);

/* The reduce guarded epilogue: result = acc * rcp(den); cast to dtype; store at
 * q_desc.offset(q_tok,q_head,dim) under active. */
void rocke_attn_unified_emit_reduce_epilogue(rocke_attn_unified_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_UNIFIED_INTERNAL_H */
