/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_helper_rocke.instances.common.attention_unified_selectors.h --
 *   C99 port of selected SELECTOR + descriptor + emit symbols from
 *   rocke/instances/common/attention_unified.py.
 *
 * PORTED SYMBOLS (this phase), mapped from their Python private names:
 *   rocke_unified_attn_select_2d_tile_size       (_select_2d_tile_size)
 *   rocke_unified_attn_select_2d_num_warps       (_select_2d_num_warps)
 *   rocke_unified_attn_select_2d_block_m_per_warp(_select_2d_block_m_per_warp)
 *   rocke_unified_attn_kv_storage_dtype          (_kv_storage_dtype)
 *   rocke_unified_attn_magic_div                 (_magic_div)
 *   rocke_unified_attn_magic_div_mod             (_magic_div_mod)
 *   rocke_unified_attn_q_descriptor              (_q_descriptor)
 *   rocke_unified_attn_paged_kv_descriptor       (_paged_kv_descriptor)
 *   rocke_unified_attn_segm_descriptors          (_segm_descriptors)
 *   rocke_unified_attn_emit_qk_score             (_emit_qk_score)
 *   rocke_unified_attn_emit_v_load               (_emit_v_load)
 *   rocke_unified_attn_physical_block_and_token  (_physical_block_and_token)
 *
 * The host-side selectors (tile_size / num_warps / block_m_per_warp /
 * kv_storage_dtype) consult the same private gate predicates the Python does
 * (_enable_combo_2d, _enable_transposed_qk_32x32, _enable_gfx942_*). Those
 * predicates are ported here too as static-internal helpers so the selectors
 * stay byte-faithful; they are not part of the public ABI.
 *
 * Every selector and gate predicate takes the target ``arch`` explicitly, as
 * its Python twin takes ``(problem, arch)``. ``arch`` must already be canonical
 * ("gfx942", not "gfx942:sramecc+"); NULL matches no arch.
 *
 * The IR-emitting members reproduce their Python rocke_b_* builder-call sequence
 * exactly (same ops, same order, same operands), binding only to rocke/ir.h plus
 * the sibling helper_*.h descriptor / magic-div surfaces.
 *
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#ifndef ROCKE_HELPER_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_H
#define ROCKE_HELPER_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_H

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/ir.h"
#include <stddef.h> /* size_t */
/* rocke_attention_tiled_2d_spec_t (the shared tiled-2D spec mirror). */
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"
/* rocke_unified_attention_problem_t -- the single merged problem struct (was
 * a separate rocke_unified_attn_problem_t here until the two were merged). */
#include "rocke/helper_helper_rocke.instances.common.attention_unified.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------- UnifiedAttentionProblem */

/* Python: @dataclass class UnifiedAttentionProblem.
 *
 * The problem struct (rocke_unified_attention_problem_t) is defined once, in
 * rocke/helper_helper_rocke.instances.common.attention_unified.h, and shared
 * by the scalar-kernel builders and the selectors below (this header used to
 * carry its own duplicate, rocke_unified_attn_problem_t, until the two were
 * merged). Only the fields the ported selectors / descriptors read are
 * exercised here, but all dataclass fields are carried so call sites can
 * populate it the same way as Python.
 *
 * `dtype` is a Literal["fp16","bf16","fp8"]-style string; it is compared with
 * strcmp exactly as the Python compares `problem.dtype`. `use_fp8` is a bool
 * field (NOT derived from dtype) -- the Python carries it separately.
 *
 * The `num_queries_per_kv` Python @property (num_query_heads // num_kv_heads,
 * raising on a non-divisible ratio) is exposed as a helper below. The selectors
 * call that helper, mirroring `problem.num_queries_per_kv`.
 */

/* Python: UnifiedAttentionProblem.num_queries_per_kv property.
 * Returns num_query_heads // num_kv_heads. On a non-divisible ratio the Python
 * raises ValueError; here the builder's sticky error is set (if `b` != NULL)
 * and 0 is returned. */
int rocke_unified_attn_num_queries_per_kv(rocke_ir_builder_t* b,
                                          const rocke_unified_attention_problem_t* p);

/* Python: _resolve_attention_arch(). The launch target: the running device
 * arch, else "gfx950" (always "gfx950" until a C device query is ported). Not a
 * selection input -- the selectors below take ``arch`` explicitly. */
const char* rocke_unified_attn_resolve_arch(void);

/* ------------------------------------------------------- host selectors */

/* Python: _select_2d_tile_size(problem, arch) -> int. */
int rocke_unified_attn_select_2d_tile_size(const rocke_unified_attention_problem_t* p,
                                           const char* arch);

/* Python: _select_2d_num_warps(problem, arch) -> int. */
int rocke_unified_attn_select_2d_num_warps(const rocke_unified_attention_problem_t* p,
                                           const char* arch);

/* Python: _select_2d_block_m_per_warp(problem, arch) -> int. */
int rocke_unified_attn_select_2d_block_m_per_warp(const rocke_unified_attention_problem_t* p,
                                                  const char* arch);

/* Python: _kv_storage_dtype(problem) -> Optional[str].
 * Returns the literal "fp8e4m3" when use_fp8, else NULL (the Python None). */
const char* rocke_unified_attn_kv_storage_dtype(const rocke_unified_attention_problem_t* p);

/* Python: _select_2d_waves_per_eu(problem, arch) -> Optional[int].
 * Writes the selected waves_per_eu into *out_wpe and returns true when the
 * Python returns a concrete int; returns false (and leaves *out_wpe untouched)
 * when the Python returns None (no override; the LLVM backend heuristic picks):
 * that is gfx1250 without a pin. A pin (p->waves_per_eu_set) wins on every arch.
 */
bool rocke_unified_attn_select_2d_waves_per_eu(const rocke_unified_attention_problem_t* p,
                                               const char* arch,
                                               int* out_wpe);

/* ------------------------------------------------------- LDS-budget resolver *
 * Selection-layer LDS accounting shared by the gfx942 admission gate and the
 * gfx950 register-PV budget resolver (mirrors the Python helpers in
 * rocke/instances/common/attention_unified.py). */

/* Python: _tiled_2d_lds_bytes -- static LDS footprint (bytes) of a tiled-2D
 * geometry. Single source of truth for the per-buffer tile arithmetic; callers
 * pick which buffers are LDS-resident (K/V slots, Q/P_lds, the block_m<=2T Q
 * alias, the OUT_STRIPE rule, the 16-bit Acc staging width). */
int rocke_unified_attn_tiled_2d_lds_bytes(int tile_size,
                                          int head_size,
                                          int block_m,
                                          int kv_elem_bytes,
                                          int k_slots,
                                          int v_slots,
                                          bool include_q_lds,
                                          bool include_p_lds,
                                          int v_pad);

/* Python: _resolve_lds_budget -- deterministically shrink an over-budget gfx950
 * register-PV 2D spec (single-buffer K, then T=64) in place until it fits the
 * arch LDS cap, using only compile-validated reductions. A strict no-op for
 * already-fitting / non-register-PV / non-gfx950 specs (returns true, spec
 * unchanged). Returns false when no validated reduction fits, filling ``reason``
 * (up to ``reason_cap`` bytes) with a diagnostic. The provider's spec-assembly
 * path calls this right after building a spec from the selectors above. */
bool rocke_unified_attn_resolve_lds_budget(rocke_attention_tiled_2d_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* ------------------------------------------------ 2D feature-gate predicates *
 * Exposed mirrors of the Python _enable_* gates that _tiled_spec_from_problem
 * consults when it builds the per-shape UnifiedAttention2DTiledSpec. The
 * provider's C-JIT build_sdpa_tiled reads these so the tiled spec it emits
 * tracks the SAME per-shape feature selection the Python selector applies
 * (instead of a fixed wide-atom default). Pure: emit no IR. */

/* Python: _enable_combo_2d(problem, arch). */
bool rocke_unified_attn_enable_combo_2d(const rocke_unified_attention_problem_t* p,
                                        const char* arch);
/* Python: _enable_transposed_qk_32x32(problem, arch). */
bool rocke_unified_attn_enable_transposed_qk_32x32(const rocke_unified_attention_problem_t* p,
                                                   const char* arch);
/* Python: _enable_mfma_32x32(problem, arch) == _enable_transposed_qk_32x32(problem, arch). */
bool rocke_unified_attn_enable_mfma_32x32(const rocke_unified_attention_problem_t* p,
                                          const char* arch);
/* Python: _enable_transposed_half_local_pv(problem, arch) == transposed_qk_32x32. */
bool rocke_unified_attn_enable_transposed_half_local_pv(const rocke_unified_attention_problem_t* p,
                                                        const char* arch);
/* Python: _enable_register_pv(problem, arch). */
bool rocke_unified_attn_enable_register_pv(const rocke_unified_attention_problem_t* p,
                                           const char* arch);

/* Python: _enable_single_batch_combo(problem, arch) -- single-batch (num_seqs==1)
 * d128/d64 prefill full-combo cohort. */
bool rocke_unified_attn_enable_single_batch_combo(const rocke_unified_attention_problem_t* p,
                                                  const char* arch);
/* Python: _enable_transposed_subflags(problem, arch) -- the no-SW transposed-softmax
 * VALU sub-flag stack (scalar_state + mask_once + mask_limit + skip_legacy_qreg)
 * for the whole no-SW transposed-32x32 cohort. */
bool rocke_unified_attn_enable_transposed_subflags(const rocke_unified_attention_problem_t* p,
                                                   const char* arch);
/* Python: _enable_d128_small_tile(problem, arch) -- d128 occupancy lever
 * (T=block_size + nw=2 -> 2 WG/CU). DEFAULT-ON for the gfx950 single-batch
 * d128 no-FP8 combo; HIPDNN_GFX950_D128_SMALL_TILE=0 force-disables. */
bool rocke_unified_attn_enable_d128_small_tile(const rocke_unified_attention_problem_t* p,
                                               const char* arch);
/* Python: _enable_softmax_mfma_interleave(problem, arch) -- d128 softmax<->MFMA
 * interleave (iglp_opt(1)) + nw=4. DEFAULT-ON for the gfx950 single-batch d128
 * no-FP8 combo; HIPDNN_GFX950_D128_SOFTMAX_INTERLEAVE=0 force-disables. */
bool rocke_unified_attn_enable_softmax_mfma_interleave(const rocke_unified_attention_problem_t* p,
                                                       const char* arch);
/* Python: _enable_v_double_buffer(problem, arch) -- short single-batch combo prefill. */
bool rocke_unified_attn_enable_v_double_buffer(const rocke_unified_attention_problem_t* p,
                                               const char* arch);
/* Python: _enable_early_v_schedule(problem, arch) -- long single-batch combo prefill. */
bool rocke_unified_attn_enable_early_v_schedule(const rocke_unified_attention_problem_t* p,
                                                const char* arch);

/* Python: _fp8_qk_loader_fits(problem, arch, *, tile_size=None, num_warps=None).
 * A ``tile_size`` / ``num_warps`` <= 0 is Python's None: the 2D selector's value
 * is used. */
bool rocke_unified_attn_fp8_qk_loader_fits(const rocke_unified_attention_problem_t* p,
                                           const char* arch,
                                           int tile_size,
                                           int num_warps);
/* Python: _enable_fp8_mfma_qk(problem, arch, *, tile_size=None, num_warps=None)
 * -- keep K as fp8 in LDS for decode / short prefill. Overrides as above. */
bool rocke_unified_attn_enable_fp8_mfma_qk(const rocke_unified_attention_problem_t* p,
                                           const char* arch,
                                           int tile_size,
                                           int num_warps);

/* True when env var ``name``, stripped and case-folded as the Python knobs read
 * it, is one of the off keywords (0, off, disable, disabled, no, false). */
bool rocke_unified_attn_env_is_off(const char* name);
/* Python: _enable_i64_kv_addr(problem) -- the paged KV cache exceeds the i32
 * buffer range. */
bool rocke_unified_attn_enable_i64_kv_addr(const rocke_unified_attention_problem_t* p);

/* --------------------------------------------- gfx942 flash / 4-warp selectors *
 * Mirrors of the Python gfx942 flash-family selectors. Several read HIPDNN_GFX942_*
 * env knobs with Python's parsing (stripped, case-folded, same keyword sets). */

/* Python: _gfx942_4warp_fast(problem, arch) -- a cohort routed to the 4-warp GQA
 * builder (D256 bf16 causal, or D128 sliding window). */
bool rocke_unified_attn_gfx942_4warp_fast(const rocke_unified_attention_problem_t* p,
                                          const char* arch);
/* Python: _enable_gfx942_fp16_flash(problem, arch). */
bool rocke_unified_attn_enable_gfx942_fp16_flash(const rocke_unified_attention_problem_t* p,
                                                 const char* arch);
/* Python: _enable_gfx942_bf16_flash(problem, arch) -- the bf16 wide-K (32x32x8)
 * transposed flash path. */
bool rocke_unified_attn_enable_gfx942_bf16_flash(const rocke_unified_attention_problem_t* p,
                                                 const char* arch);
/* Python: _enable_gfx942_d128_smalltile_doublek(problem, arch). Reads
 * HIPDNN_GFX942_D128_SMALLTILE_DK. */
bool rocke_unified_attn_enable_gfx942_d128_smalltile_doublek(
    const rocke_unified_attention_problem_t* p, const char* arch);
/* Python: _gfx942_bf16_wide_tile_size(problem, arch). */
int rocke_unified_attn_gfx942_bf16_wide_tile_size(const rocke_unified_attention_problem_t* p,
                                                  const char* arch);
/* Python: _gfx942_bf16_wide_geometry(problem, arch) -> (num_warps,
 * use_k_single_buffer). Either out pointer may be NULL. */
void rocke_unified_attn_gfx942_bf16_wide_geometry(const rocke_unified_attention_problem_t* p,
                                                  const char* arch,
                                                  int* num_warps,
                                                  bool* use_k_single_buffer);
/* Python: _gfx942_bf16_wide_use_cfvst(problem). Reads HIPDNN_GFX942_BF16_CFVST. */
bool rocke_unified_attn_gfx942_bf16_wide_use_cfvst(const rocke_unified_attention_problem_t* p);
/* Python: _gfx942_flash_use_cfvst(problem). Reads HIPDNN_GFX942_FLASH_WIDE. */
bool rocke_unified_attn_gfx942_flash_use_cfvst(const rocke_unified_attention_problem_t* p);
/* Python: _gfx942_flash_use_single_buffer(problem). */
bool rocke_unified_attn_gfx942_flash_use_single_buffer(const rocke_unified_attention_problem_t* p);
/* Python: _enable_gfx942_flash_mask_limit(problem, arch). Reads
 * HIPDNN_GFX942_FLASH_MLIM. */
bool rocke_unified_attn_enable_gfx942_flash_mask_limit(const rocke_unified_attention_problem_t* p,
                                                       const char* arch);
/* Python: _enable_gfx942_flash_k_sliced_ring(problem, arch). Reads
 * HIPDNN_GFX942_K_SLICED_RING. */
bool rocke_unified_attn_enable_gfx942_flash_k_sliced_ring(
    const rocke_unified_attention_problem_t* p, const char* arch);
/* Python: _select_gfx942_flash_ring_depth(problem, arch). */
int rocke_unified_attn_select_gfx942_flash_ring_depth(const rocke_unified_attention_problem_t* p,
                                                      const char* arch);
/* Python: _select_gfx942_flash_k_slice_hd(problem). Reads HIPDNN_GFX942_K_SLICE_HD. */
int rocke_unified_attn_select_gfx942_flash_k_slice_hd(const rocke_unified_attention_problem_t* p);
/* Python: _select_gfx942_flash_num_warps(problem, arch). Reads
 * HIPDNN_GFX942_FLASH_WIDE. */
int rocke_unified_attn_select_gfx942_flash_num_warps(const rocke_unified_attention_problem_t* p,
                                                     const char* arch);
/* Python: _enable_gfx942_flash_q_direct(problem, arch). */
bool rocke_unified_attn_enable_gfx942_flash_q_direct(const rocke_unified_attention_problem_t* p,
                                                     const char* arch);
/* Python: _gfx942_flash_kv_cache_policy(problem). */
const char*
    rocke_unified_attn_gfx942_flash_kv_cache_policy(const rocke_unified_attention_problem_t* p);
/* Python: _enable_gfx942_flash_k_sliced_ldsseq(problem, arch). Reads
 * HIPDNN_GFX942_K_LDSSEQ. */
bool rocke_unified_attn_enable_gfx942_flash_k_sliced_ldsseq(
    const rocke_unified_attention_problem_t* p, const char* arch);

/* Python: attention_spec_builder._tiled_spec_from_problem(problem, arch) ->
 * rocke_attention_tiled_2d_spec_t. Runs the same selectors and gate predicates
 * in the same order: the gfx942 bf16 / fp16 flash builders, the gfx950 generic
 * builder, and the generic builder every other arch shares. Not ported: the
 * gfx1250 branch (returns the default spec, as does a NULL arch), the gfx950
 * D256 fast-route overrides and the softmax-interleave fields.
 *
 * Python's attention_unified._tiled_spec_from_problem also validates the spec
 * (the spec class __post_init__) and runs _resolve_lds_budget. Callers do both
 * here: rocke_attention_tiled_2d_spec_validate for a gfx942 spec, then
 * rocke_unified_attn_resolve_lds_budget. */
rocke_attention_tiled_2d_spec_t
    rocke_unified_attn_tiled_spec_from_problem(const rocke_unified_attention_problem_t* p,
                                               const char* arch);

/* ----------------------------------------------------------- magic div */

/* Python: _magic_div(b, dividend, divisor) -> Value.
 * dividend // divisor via the CK-Tile mul-hi magic sequence. */
rocke_value_t*
    rocke_unified_attn_magic_div(rocke_ir_builder_t* b, rocke_value_t* dividend, int divisor);

/* Python: _magic_div_mod(b, dividend, divisor) -> (quotient, remainder).
 * quotient = magic_div(...); remainder = dividend - quotient * divisor.
 * Writes *out_quotient and *out_remainder; returns true on success. */
bool rocke_unified_attn_magic_div_mod(rocke_ir_builder_t* b,
                                      rocke_value_t* dividend,
                                      int divisor,
                                      rocke_value_t** out_quotient,
                                      rocke_value_t** out_remainder);

/* --------------------------------------------------------- descriptors */

/* Python: _q_descriptor(p) -> TensorDescriptor.
 * Element-unit Q/output descriptor with coords (token, head, dim) and lengths
 * [max_seqlen_q + 1, num_query_heads, head_size]. Arena-owned; NULL on error. */
rocke_tensor_descriptor_t*
    rocke_unified_attn_q_descriptor(rocke_ir_builder_t* b,
                                    const rocke_unified_attention_problem_t* p);

/* Python: @dataclass class PagedKvDescriptor (helpers/attention.py).
 * Address helper for [num_blocks, block_size, num_kv_heads, head] KV. The
 * offset() member is reproduced by rocke_unified_attn_paged_kv_offset below. */
typedef struct rocke_unified_attn_paged_kv_descriptor
{
    int block_size;
    int stride_0;
    int stride_1;
    int stride_2;
    int stride_3;
} rocke_unified_attn_paged_kv_descriptor_t;

/* Python: _paged_kv_descriptor(p) -> PagedKvDescriptor.
 * Element-unit paged-KV descriptor for the scalar kernels. */
rocke_unified_attn_paged_kv_descriptor_t
    rocke_unified_attn_paged_kv_descriptor(const rocke_unified_attention_problem_t* p);

/* Python: PagedKvDescriptor.offset(b, physical_block, token_in_block, kv_head,
 *                                  dim) -> Value.
 * off = physical_block*s0 + token_in_block*s1 + kv_head*s2 + dim*s3, emitted as
 * the same mul/add ladder as the Python. */
rocke_value_t* rocke_unified_attn_paged_kv_offset(rocke_ir_builder_t* b,
                                                  const rocke_unified_attn_paged_kv_descriptor_t* d,
                                                  rocke_value_t* physical_block,
                                                  rocke_value_t* token_in_block,
                                                  rocke_value_t* kv_head,
                                                  rocke_value_t* dim);

/* Python: _segm_descriptors(p, num_segments) -> (segm_ml, segm_output).
 * segm_ml     : [max_seqlen_q+1, num_query_heads, num_segments]
 * segm_output : [max_seqlen_q+1, num_query_heads, num_segments, head_size]
 * Writes both arena-owned descriptors; returns true on success. */
bool rocke_unified_attn_segm_descriptors(rocke_ir_builder_t* b,
                                         const rocke_unified_attention_problem_t* p,
                                         int num_segments,
                                         rocke_tensor_descriptor_t** out_ml,
                                         rocke_tensor_descriptor_t** out_output);

/* ------------------------------------------------------- IR emit helpers */

/* Python: _physical_block_and_token(b, p, block_tables, bt_stride, seq_idx, kpos)
 *   -> (physical, token_in_block).
 * (block_idx, token_in_block) = magic_div_mod(kpos, block_size);
 * physical = global_load_i32(block_tables, seq_idx*bt_stride + block_idx).
 * bt_stride is the block_table_stride kernarg (the table's row stride).
 * Writes *out_physical and *out_token_in_block; returns true on success. */
bool rocke_unified_attn_physical_block_and_token(rocke_ir_builder_t* b,
                                                 const rocke_unified_attention_problem_t* p,
                                                 rocke_value_t* block_tables,
                                                 rocke_value_t* bt_stride,
                                                 rocke_value_t* seq_idx,
                                                 rocke_value_t* kpos,
                                                 rocke_value_t** out_physical,
                                                 rocke_value_t** out_token_in_block);

/* Python: _emit_qk_score(b, p, dtype, query, key, block_tables, bt_stride,
 *                        seq_idx, q_tok, q_head, kv_head, kpos, scale,
 *                        rcp_ln2) -> Value.
 * Per-(query_token, query_head) QK dot product for the scalar kernels (vec8 main
 * fold + scalar tail), scaled by scale*rcp_ln2. `dtype` is the IR element type
 * (the Python passes a Type). */
rocke_value_t* rocke_unified_attn_emit_qk_score(rocke_ir_builder_t* b,
                                                const rocke_unified_attention_problem_t* p,
                                                const rocke_type_t* dtype,
                                                rocke_value_t* query,
                                                rocke_value_t* key,
                                                rocke_value_t* block_tables,
                                                rocke_value_t* bt_stride,
                                                rocke_value_t* seq_idx,
                                                rocke_value_t* q_tok,
                                                rocke_value_t* q_head,
                                                rocke_value_t* kv_head,
                                                rocke_value_t* kpos,
                                                rocke_value_t* scale,
                                                rocke_value_t* rcp_ln2);

/* Python: _emit_v_load(b, p, dtype, value, block_tables, bt_stride, seq_idx,
 *                      kv_head, kpos, dim) -> Value.
 * Loads V[physical, token_in_block, kv_head, dim] and casts to f32. */
rocke_value_t* rocke_unified_attn_emit_v_load(rocke_ir_builder_t* b,
                                              const rocke_unified_attention_problem_t* p,
                                              const rocke_type_t* dtype,
                                              rocke_value_t* value,
                                              rocke_value_t* block_tables,
                                              rocke_value_t* bt_stride,
                                              rocke_value_t* seq_idx,
                                              rocke_value_t* kv_head,
                                              rocke_value_t* kpos,
                                              rocke_value_t* dim);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_UNIFIED_SELECTORS_H */
