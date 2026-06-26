/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_wmma_attention_common.h -- C99 port of the shared
 * building blocks for the gfx1250 wave32 WMMA attention kernels
 * rocke/instances/gfx1250/_wmma_attention_common.py.
 *
 * INTERNAL header: these helpers are consumed by the attention_tiled_2d and
 * attention_tiled_3d instance ports, not by external callers. Each helper
 * reproduces its Python counterpart's rocke_b_* builder-call sequence
 * byte-faithfully (same ops, same order, same operands) so the emitted IR is
 * byte-identical to the Python lowerer.
 *
 * The Python ``kv_desc`` (PagedKvDescriptor) is passed here as an explicit
 * rocke_kv_desc_t (the four strides); kv_desc.offset() is reproduced by
 * rocke_g1250_kv_offset. The Python ``phys_block`` callable is a
 * rocke_phys_block_fn_t function pointer + opaque ctx so the 2D kernel (one
 * cached block per tile) and the 3D kernel (per-token lookup) can each supply
 * their own without changing the emitted code.
 */
#ifndef ROCKE_INSTANCE_GFX1250_WMMA_ATTENTION_COMMON_H
#define ROCKE_INSTANCE_GFX1250_WMMA_ATTENTION_COMMON_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arch_target.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_G1250_WMMA_OP_ID "wmma_gfx1250_f32_16x16x32_bf16"
#define ROCKE_G1250_WAVE 32
#define ROCKE_G1250_BLOCK_M 16
#define ROCKE_G1250_WMMA_N 16
#define ROCKE_G1250_WMMA_K 32
#define ROCKE_G1250_HEAD_SIZE 64

/* PagedKvDescriptor analogue (the four strides). */
typedef struct rocke_kv_desc
{
    int block_size;
    int stride_0;
    int stride_1;
    int stride_2;
    int stride_3;
} rocke_kv_desc_t;

/* kv_desc.offset(b, physical_block, token_in_block, kv_head, dim). */
rocke_value_t* rocke_g1250_kv_offset(rocke_ir_builder_t* b,
                                     const rocke_kv_desc_t* d,
                                     rocke_value_t* physical_block,
                                     rocke_value_t* token_in_block,
                                     rocke_value_t* kv_head,
                                     rocke_value_t* dim);

/* phys_block(tok) -> physical block id. ctx is the caller's closure state. */
typedef rocke_value_t* (*rocke_phys_block_fn_t)(rocke_ir_builder_t* b,
                                                rocke_value_t* tok,
                                                void* ctx);

/* kv_storage_ir(kv_storage_dtype): bf16 (NULL/"bf16") or fp8e4m3. NULL on bad. */
const rocke_type_t* rocke_g1250_kv_storage_ir(const char* kv_storage_dtype);

/* check_wmma_arch(arch) -> ok; writes reason. */
bool rocke_g1250_check_wmma_arch(const char* arch, char* reason, size_t reason_cap);

/* resolve_wmma(arch): returns the op (NULL on failure) and, via out-params, the
 * a/c layout maps and a/c frag lengths. */
const rocke_mma_op_t* rocke_g1250_resolve_wmma(rocke_ir_builder_t* b,
                                               const char* arch,
                                               const rocke_layout_map_t** a_layout_out,
                                               const rocke_layout_map_t** c_layout_out,
                                               int* a_frag_out,
                                               int* c_frag_out);

/* load_kv16(b, ptr, base, scale, kv_dtype, out_dtype) -> <16 x out_dtype>. */
rocke_value_t* rocke_g1250_load_kv16(rocke_ir_builder_t* b,
                                     rocke_value_t* ptr,
                                     rocke_value_t* base,
                                     rocke_value_t* scale,
                                     const rocke_type_t* kv_dtype,
                                     const rocke_type_t* out_dtype);

/* load_q_frags(...) -> writes head_size/WMMA_K frags into out_frags[] (caller
 * sizes the array). */
void rocke_g1250_load_q_frags(rocke_ir_builder_t* b,
                              rocke_value_t* query,
                              rocke_value_t* q_addr_row_base,
                              rocke_value_t* half_k,
                              rocke_value_t* q_valid,
                              int head_size,
                              int a_frag,
                              const rocke_type_t* dtype,
                              rocke_value_t** out_frags);

/* compute_qk_scores(...) -> writes scores[0], scores[1]. */
void rocke_g1250_compute_qk_scores(rocke_ir_builder_t* b,
                                   rocke_value_t* const* q_frags,
                                   rocke_value_t* key,
                                   const rocke_kv_desc_t* kv_desc,
                                   rocke_value_t* tile_base,
                                   rocke_value_t* lane_row,
                                   rocke_value_t* half_k,
                                   rocke_value_t* kv_head_idx,
                                   int block_size,
                                   int head_size,
                                   const rocke_type_t* kv_dtype,
                                   rocke_value_t* k_scale,
                                   const rocke_type_t* dtype,
                                   int c_frag,
                                   rocke_phys_block_fn_t phys_block,
                                   void* phys_ctx,
                                   int spacing,
                                   rocke_value_t* out_scores[2]);

/* softmax_row_update(...) -> m_new/l_new/alpha/p[0..1] via out-params. */
void rocke_g1250_softmax_row_update(rocke_ir_builder_t* b,
                                    rocke_value_t* m_prev,
                                    rocke_value_t* l_prev,
                                    rocke_value_t* const srs[2],
                                    rocke_value_t* neg_inf,
                                    rocke_value_t* zero_f,
                                    bool use_dpp,
                                    rocke_value_t** m_new_out,
                                    rocke_value_t** l_new_out,
                                    rocke_value_t** alpha_out,
                                    rocke_value_t* p_out[2]);

/* stage_v_tile(...) into V_lds[lane, dim]. */
void rocke_g1250_stage_v_tile(rocke_ir_builder_t* b,
                              rocke_value_t* V_lds,
                              rocke_value_t* value,
                              const rocke_kv_desc_t* kv_desc,
                              rocke_value_t* kv_head_idx,
                              rocke_value_t* tile_base,
                              rocke_value_t* lane,
                              int block_size,
                              int head_size,
                              const rocke_type_t* kv_dtype,
                              rocke_value_t* v_scale,
                              const rocke_type_t* dtype,
                              rocke_phys_block_fn_t phys_block,
                              void* phys_ctx);

/* stage_v_tile_buf(...) into V_lds[buf_idx, lane, dim]. */
void rocke_g1250_stage_v_tile_buf(rocke_ir_builder_t* b,
                                  rocke_value_t* V_lds,
                                  rocke_value_t* buf_idx,
                                  rocke_value_t* value,
                                  const rocke_kv_desc_t* kv_desc,
                                  rocke_value_t* kv_head_idx,
                                  rocke_value_t* tile_base,
                                  rocke_value_t* lane,
                                  int block_size,
                                  int head_size,
                                  const rocke_type_t* kv_dtype,
                                  rocke_value_t* v_scale,
                                  const rocke_type_t* dtype,
                                  rocke_phys_block_fn_t phys_block,
                                  void* phys_ctx);

/* stage_v_tile_transposed(...) into dim-major V_lds_T[(buf,)dim,lane].
 * buf_idx NULL => 2D [dim, lane] indexing. */
void rocke_g1250_stage_v_tile_transposed(rocke_ir_builder_t* b,
                                         rocke_value_t* V_lds_T,
                                         rocke_value_t* value,
                                         const rocke_kv_desc_t* kv_desc,
                                         rocke_value_t* kv_head_idx,
                                         rocke_value_t* tile_base,
                                         rocke_value_t* lane,
                                         int block_size,
                                         int head_size,
                                         const rocke_type_t* kv_dtype,
                                         rocke_value_t* v_scale,
                                         const rocke_type_t* dtype,
                                         rocke_phys_block_fn_t phys_block,
                                         void* phys_ctx,
                                         rocke_value_t* buf_idx);

/* compute_pv(...): scalar a_map gather of P + V; accumulate into accs[] (length
 * head_size/WMMA_N), result written back into accs[]. v_extra_idx/p_extra_idx
 * NULL => 2D indexing. */
void rocke_g1250_compute_pv(rocke_ir_builder_t* b,
                            rocke_value_t* P_lds,
                            rocke_value_t* V_lds,
                            rocke_value_t** accs,
                            const rocke_layout_map_t* a_map,
                            rocke_value_t* lane,
                            rocke_value_t* lane_row,
                            rocke_value_t* col,
                            int a_frag,
                            int c_frag,
                            int head_size,
                            const rocke_type_t* dtype,
                            rocke_value_t* v_extra_idx,
                            rocke_value_t* p_extra_idx,
                            int spacing);

/* compute_pv_wide(...): wide LDS reads (V_lds_T dim-major). */
void rocke_g1250_compute_pv_wide(rocke_ir_builder_t* b,
                                 rocke_value_t* P_lds,
                                 rocke_value_t* V_lds_T,
                                 rocke_value_t** accs,
                                 const rocke_layout_map_t* a_map,
                                 rocke_value_t* lane,
                                 rocke_value_t* lane_row,
                                 int a_frag,
                                 int head_size,
                                 const rocke_type_t* dtype,
                                 rocke_value_t* v_extra_idx,
                                 rocke_value_t* p_extra_idx,
                                 int spacing);

/* compute_pv_dstr(...): gfx1250 ds_load_tr16_b128 transpose-LDS read path. */
void rocke_g1250_compute_pv_dstr(rocke_ir_builder_t* b,
                                 rocke_value_t* P_lds,
                                 rocke_value_t* V_lds,
                                 rocke_value_t** accs,
                                 const rocke_layout_map_t* a_map,
                                 rocke_value_t* lane,
                                 rocke_value_t* lane_row,
                                 int a_frag,
                                 int head_size,
                                 const rocke_type_t* dtype,
                                 rocke_value_t* v_extra_idx,
                                 rocke_value_t* p_extra_idx,
                                 int spacing);

/* compute_pv_from_probs(...): register-resident probs (no P_lds round-trip).
 * ps0/ps1 have length c_frag. */
void rocke_g1250_compute_pv_from_probs(rocke_ir_builder_t* b,
                                       rocke_value_t* const* ps0,
                                       rocke_value_t* const* ps1,
                                       rocke_value_t* V_lds,
                                       rocke_value_t** accs,
                                       const rocke_layout_map_t* a_map,
                                       rocke_value_t* lane,
                                       rocke_value_t* col,
                                       int a_frag,
                                       int c_frag,
                                       int head_size,
                                       const rocke_type_t* dtype,
                                       rocke_value_t* v_extra_idx,
                                       int spacing);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_WMMA_ATTENTION_COMMON_H */
