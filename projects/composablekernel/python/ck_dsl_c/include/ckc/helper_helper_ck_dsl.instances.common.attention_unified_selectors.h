/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_helper_ck_dsl.instances.common.attention_unified_selectors.h --
 *   C99 port of selected SELECTOR + descriptor + emit symbols from
 *   ck_dsl/instances/common/attention_unified.py.
 *
 * PORTED SYMBOLS (this phase), mapped from their Python private names:
 *   ckc_unified_attn_select_2d_tile_size       (_select_2d_tile_size)
 *   ckc_unified_attn_select_2d_num_warps       (_select_2d_num_warps)
 *   ckc_unified_attn_select_2d_block_m_per_warp(_select_2d_block_m_per_warp)
 *   ckc_unified_attn_kv_storage_dtype          (_kv_storage_dtype)
 *   ckc_unified_attn_magic_div                 (_magic_div)
 *   ckc_unified_attn_magic_div_mod             (_magic_div_mod)
 *   ckc_unified_attn_q_descriptor              (_q_descriptor)
 *   ckc_unified_attn_paged_kv_descriptor       (_paged_kv_descriptor)
 *   ckc_unified_attn_segm_descriptors          (_segm_descriptors)
 *   ckc_unified_attn_emit_qk_score             (_emit_qk_score)
 *   ckc_unified_attn_emit_v_load               (_emit_v_load)
 *   ckc_unified_attn_physical_block_and_token  (_physical_block_and_token)
 *
 * The host-side selectors (tile_size / num_warps / block_m_per_warp /
 * kv_storage_dtype) consult the same private gate predicates the Python does
 * (_enable_combo_2d, _enable_transposed_qk_32x32, _enable_gfx942_*). Those
 * predicates are ported here too as static-internal helpers so the selectors
 * stay byte-faithful; they are not part of the public ABI.
 *
 * The IR-emitting members reproduce their Python ckc_b_* builder-call sequence
 * exactly (same ops, same order, same operands), binding only to ckc/ir.h plus
 * the sibling helper_*.h descriptor / magic-div surfaces.
 *
 * Lifetime: every emitted node is arena-owned (ckc_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#ifndef CKC_HELPER_HELPER_CK_DSL_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_H
#define CKC_HELPER_HELPER_CK_DSL_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_H

#include "ckc/ir.h"
#include "ckc/helper_ck_dsl.helpers.transforms.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------- UnifiedAttentionProblem */

/* Python: @dataclass class UnifiedAttentionProblem.
 *
 * Plain value struct (host-side selector input). Only the fields the ported
 * selectors / descriptors read are exercised, but all dataclass fields are
 * carried so call sites can populate it the same way as Python.
 *
 * `dtype` is a Literal["fp16","bf16","fp8"]-style string; it is compared with
 * strcmp exactly as the Python compares `problem.dtype`. `use_fp8` is a bool
 * field (NOT derived from dtype) -- the Python carries it separately.
 *
 * The `num_queries_per_kv` Python @property (num_query_heads // num_kv_heads,
 * raising on a non-divisible ratio) is exposed as a helper below. The selectors
 * call that helper, mirroring `problem.num_queries_per_kv`.
 */
typedef struct ckc_unified_attn_problem
{
    int total_q;
    int num_seqs;
    int num_query_heads;
    int num_kv_heads;
    int head_size;
    int block_size;
    int max_seqlen_q;
    int max_seqlen_k;
    const char* dtype;   /* "fp16" / "bf16" / "fp8" ... */
    const char* q_dtype; /* or NULL */
    int sliding_window;  /* default 0 */
    double softcap;      /* default 0.0 */
    bool use_sinks;      /* default false */
    bool use_alibi;      /* default false */
    bool use_qq_bias;    /* default false */
    bool use_fp8;        /* default false */
    int num_sms;         /* default 120 */
    /* waves_per_eu / compile_backend are not read by the ported selectors. */
    int num_kv_blocks; /* default 0 ("unknown") */
} ckc_unified_attn_problem_t;

/* Python: UnifiedAttentionProblem.num_queries_per_kv property.
 * Returns num_query_heads // num_kv_heads. On a non-divisible ratio the Python
 * raises ValueError; here the builder's sticky error is set (if `b` != NULL)
 * and 0 is returned. */
int ckc_unified_attn_num_queries_per_kv(ckc_ir_builder_t* b, const ckc_unified_attn_problem_t* p);

/* Python: _resolve_attention_arch() override hook. The Python memoizes the
 * device arch (falling back to "gfx950"); this lets a host pin the arch the
 * selectors resolve to (NULL restores the "gfx950" fallback). Pass a string
 * with static/long lifetime -- it is stored by pointer, not copied. */
void ckc_unified_attn_set_resolved_arch(const char* arch);

/* ------------------------------------------------------- host selectors */

/* Python: _select_2d_tile_size(problem) -> int. */
int ckc_unified_attn_select_2d_tile_size(const ckc_unified_attn_problem_t* p);

/* Python: _select_2d_num_warps(problem) -> int. */
int ckc_unified_attn_select_2d_num_warps(const ckc_unified_attn_problem_t* p);

/* Python: _select_2d_block_m_per_warp(problem) -> int. */
int ckc_unified_attn_select_2d_block_m_per_warp(const ckc_unified_attn_problem_t* p);

/* Python: _kv_storage_dtype(problem) -> Optional[str].
 * Returns the literal "fp8e4m3" when use_fp8, else NULL (the Python None). */
const char* ckc_unified_attn_kv_storage_dtype(const ckc_unified_attn_problem_t* p);

/* ----------------------------------------------------------- magic div */

/* Python: _magic_div(b, dividend, divisor) -> Value.
 * dividend // divisor via the CK-Tile mul-hi magic sequence. */
ckc_value_t* ckc_unified_attn_magic_div(ckc_ir_builder_t* b, ckc_value_t* dividend, int divisor);

/* Python: _magic_div_mod(b, dividend, divisor) -> (quotient, remainder).
 * quotient = magic_div(...); remainder = dividend - quotient * divisor.
 * Writes *out_quotient and *out_remainder; returns true on success. */
bool ckc_unified_attn_magic_div_mod(ckc_ir_builder_t* b,
                                    ckc_value_t* dividend,
                                    int divisor,
                                    ckc_value_t** out_quotient,
                                    ckc_value_t** out_remainder);

/* --------------------------------------------------------- descriptors */

/* Python: _q_descriptor(p) -> TensorDescriptor.
 * Element-unit Q/output descriptor with coords (token, head, dim) and lengths
 * [max_seqlen_q + 1, num_query_heads, head_size]. Arena-owned; NULL on error. */
ckc_tensor_descriptor_t* ckc_unified_attn_q_descriptor(ckc_ir_builder_t* b,
                                                       const ckc_unified_attn_problem_t* p);

/* Python: @dataclass class PagedKvDescriptor (helpers/attention.py).
 * Address helper for [num_blocks, block_size, num_kv_heads, head] KV. The
 * offset() member is reproduced by ckc_unified_attn_paged_kv_offset below. */
typedef struct ckc_unified_attn_paged_kv_descriptor
{
    int block_size;
    int stride_0;
    int stride_1;
    int stride_2;
    int stride_3;
} ckc_unified_attn_paged_kv_descriptor_t;

/* Python: _paged_kv_descriptor(p) -> PagedKvDescriptor.
 * Element-unit paged-KV descriptor for the scalar kernels. */
ckc_unified_attn_paged_kv_descriptor_t
ckc_unified_attn_paged_kv_descriptor(const ckc_unified_attn_problem_t* p);

/* Python: PagedKvDescriptor.offset(b, physical_block, token_in_block, kv_head,
 *                                  dim) -> Value.
 * off = physical_block*s0 + token_in_block*s1 + kv_head*s2 + dim*s3, emitted as
 * the same mul/add ladder as the Python. */
ckc_value_t* ckc_unified_attn_paged_kv_offset(ckc_ir_builder_t* b,
                                              const ckc_unified_attn_paged_kv_descriptor_t* d,
                                              ckc_value_t* physical_block,
                                              ckc_value_t* token_in_block,
                                              ckc_value_t* kv_head,
                                              ckc_value_t* dim);

/* Python: _segm_descriptors(p, num_segments) -> (segm_ml, segm_output).
 * segm_ml     : [max_seqlen_q+1, num_query_heads, num_segments]
 * segm_output : [max_seqlen_q+1, num_query_heads, num_segments, head_size]
 * Writes both arena-owned descriptors; returns true on success. */
bool ckc_unified_attn_segm_descriptors(ckc_ir_builder_t* b,
                                       const ckc_unified_attn_problem_t* p,
                                       int num_segments,
                                       ckc_tensor_descriptor_t** out_ml,
                                       ckc_tensor_descriptor_t** out_output);

/* ------------------------------------------------------- IR emit helpers */

/* Python: _physical_block_and_token(b, p, block_tables, seq_idx, kpos)
 *   -> (physical, token_in_block).
 * (block_idx, token_in_block) = magic_div_mod(kpos, block_size);
 * max_blocks = ceil(max_seqlen_k / block_size);
 * physical = global_load_i32(block_tables, seq_idx*max_blocks + block_idx).
 * Writes *out_physical and *out_token_in_block; returns true on success. */
bool ckc_unified_attn_physical_block_and_token(ckc_ir_builder_t* b,
                                               const ckc_unified_attn_problem_t* p,
                                               ckc_value_t* block_tables,
                                               ckc_value_t* seq_idx,
                                               ckc_value_t* kpos,
                                               ckc_value_t** out_physical,
                                               ckc_value_t** out_token_in_block);

/* Python: _emit_qk_score(b, p, dtype, query, key, block_tables, seq_idx, q_tok,
 *                        q_head, kv_head, kpos, scale, rcp_ln2) -> Value.
 * Per-(query_token, query_head) QK dot product for the scalar kernels (vec8 main
 * fold + scalar tail), scaled by scale*rcp_ln2. `dtype` is the IR element type
 * (the Python passes a Type). */
ckc_value_t* ckc_unified_attn_emit_qk_score(ckc_ir_builder_t* b,
                                            const ckc_unified_attn_problem_t* p,
                                            const ckc_type_t* dtype,
                                            ckc_value_t* query,
                                            ckc_value_t* key,
                                            ckc_value_t* block_tables,
                                            ckc_value_t* seq_idx,
                                            ckc_value_t* q_tok,
                                            ckc_value_t* q_head,
                                            ckc_value_t* kv_head,
                                            ckc_value_t* kpos,
                                            ckc_value_t* scale,
                                            ckc_value_t* rcp_ln2);

/* Python: _emit_v_load(b, p, dtype, value, block_tables, seq_idx, kv_head, kpos,
 *                      dim) -> Value.
 * Loads V[physical, token_in_block, kv_head, dim] and casts to f32. */
ckc_value_t* ckc_unified_attn_emit_v_load(ckc_ir_builder_t* b,
                                          const ckc_unified_attn_problem_t* p,
                                          const ckc_type_t* dtype,
                                          ckc_value_t* value,
                                          ckc_value_t* block_tables,
                                          ckc_value_t* seq_idx,
                                          ckc_value_t* kv_head,
                                          ckc_value_t* kpos,
                                          ckc_value_t* dim);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_HELPER_CK_DSL_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_H */
