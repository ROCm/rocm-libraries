/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gemm_mn_operand.h -- C++ mirror of
 * rocke/instances/common/gemm_mn_operand.py: MN-contiguous GEMM operands
 * (layouts RRR / CRR / CCR) on the gfx950 alias-scoped DTL ring.
 *
 *   MnLayout                      rocke_gemm_mn_layout_t + rocke_gemm_mn_layout_init
 *   mn_layout_letters             rocke_gemm_mn_layout_letters
 *   default_mn_chunk              rocke_gemm_default_mn_chunk
 *   uses_mn_dtl / spec_mn_layouts rocke_gemm_uses_mn_dtl / rocke_gemm_spec_mn_layouts
 *   mn_layout_error               rocke_gemm_mn_layout_error
 *   mn_lds_extra_bytes            rocke_gemm_mn_lds_extra_bytes
 *   emit_dtl_voffs / emit_dtl_lane_k_mn / emit_read_lane_elems / emit_tr_fragment
 *
 * Scope: the DTL-ring path (dtl_stages >= 2). The VGPR-staged MN operand
 * loader of the other bodies (mn_vec, vp_mn_*) is implemented by the Python
 * engine only: rocke_gemm_mn_layout_error refuses those bodies.
 */
#ifndef ROCKE_INSTANCE_GEMM_MN_OPERAND_H
#define ROCKE_INSTANCE_GEMM_MN_OPERAND_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_GEMM_MN_BLOCK_BYTES 1024
#define ROCKE_GEMM_MN_MAX_BITS 16

/* Python MnLayout (frozen dataclass; W, R, row_bits, blk_bits derived). */
typedef struct rocke_gemm_mn_layout
{
    int tile_mn;
    int tile_k;
    int block_size;
    int wave_size;
    int warp_ext; /* MN elements one wave's fragments span */
    int pad; /* bytes after every 1 KiB block */
    int chunk; /* MN chunk width (0: default_mn_chunk) */
    bool sub_perm;
    int W;
    int R;
    int row_bits[ROCKE_GEMM_MN_MAX_BITS];
    int n_row_bits;
    int blk_bits[ROCKE_GEMM_MN_MAX_BITS];
    int n_blk_bits;
} rocke_gemm_mn_layout_t;

/* MnLayout(...) incl. __post_init__; false with Python's ValueError message. */
bool rocke_gemm_mn_layout_init(rocke_gemm_mn_layout_t* lay,
                               int tile_mn,
                               int tile_k,
                               int block_size,
                               int wave_size,
                               int warp_ext,
                               int pad,
                               int chunk,
                               bool sub_perm,
                               char* reason,
                               size_t cap);

int rocke_gemm_mn_n_waves(const rocke_gemm_mn_layout_t* l);
int rocke_gemm_mn_block_bytes(const rocke_gemm_mn_layout_t* l);
int rocke_gemm_mn_blocks_per_chunk(const rocke_gemm_mn_layout_t* l);
int rocke_gemm_mn_n_blocks(const rocke_gemm_mn_layout_t* l);
int rocke_gemm_mn_n_pass(const rocke_gemm_mn_layout_t* l);
int rocke_gemm_mn_stage_bytes(const rocke_gemm_mn_layout_t* l);
int rocke_gemm_mn_wave_bytes(const rocke_gemm_mn_layout_t* l);
int rocke_gemm_mn_pass_bytes(const rocke_gemm_mn_layout_t* l);
int rocke_gemm_mn_k_contrib(const rocke_gemm_mn_layout_t* l, int k);
int rocke_gemm_mn_mn_contrib(const rocke_gemm_mn_layout_t* l, int mn);
int rocke_gemm_mn_read_imm_bytes(const rocke_gemm_mn_layout_t* l, int idx, int r, int kk);

/* default_mn_chunk; 0 with `reason` when none exists. */
int rocke_gemm_default_mn_chunk(int tile_mn, int tile_k, int warp_ext, char* reason, size_t cap);

/* mn_layout_letters(layout): false for an unsupported layout string. */
bool rocke_gemm_mn_layout_letters(const char* layout, bool* a_mn, bool* b_mn);

/* uses_mn_dtl(spec) */
bool rocke_gemm_uses_mn_dtl(const rocke_gemm_universal_spec_t* spec);

/* spec_mn_layouts(spec): has_a / has_b false for a K-contiguous operand (and
 * for both without the DTL ring). False with the ValueError message. */
bool rocke_gemm_spec_mn_layouts(const rocke_gemm_universal_spec_t* spec,
                                rocke_gemm_mn_layout_t* la,
                                bool* has_a,
                                rocke_gemm_mn_layout_t* lb,
                                bool* has_b,
                                char* reason,
                                size_t cap);

/* mn_layout_error(spec, arch): true (with the reason) when the spec cannot use
 * its MN-contiguous layout; false when it can or the layout is RCR. */
bool rocke_gemm_mn_layout_error(const rocke_gemm_universal_spec_t* spec,
                                const char* arch,
                                char* reason,
                                size_t cap);

/* mn_lds_extra_bytes(spec) */
int rocke_gemm_mn_lds_extra_bytes(const rocke_gemm_universal_spec_t* spec);

/* emit_dtl_voffs(b, lay, tid, ld): one i32 element offset per pass in out[]. */
void rocke_gemm_mn_emit_dtl_voffs(rocke_ir_builder_t* b,
                                  const rocke_gemm_mn_layout_t* lay,
                                  rocke_value_t* tid,
                                  rocke_value_t* ld,
                                  rocke_value_t** out);

/* emit_dtl_lane_k_mn(b, lay, tid): per pass (k, mn) in k_out[] / mn_out[]. */
void rocke_gemm_mn_emit_dtl_lane_k_mn(rocke_ir_builder_t* b,
                                      const rocke_gemm_mn_layout_t* lay,
                                      rocke_value_t* tid,
                                      rocke_value_t** k_out,
                                      rocke_value_t** mn_out);

/* emit_read_lane_elems(b, lay, lane, warp_idx) */
rocke_value_t* rocke_gemm_mn_emit_read_lane_elems(rocke_ir_builder_t* b,
                                                  const rocke_gemm_mn_layout_t* lay,
                                                  rocke_value_t* lane,
                                                  rocke_value_t* warp_idx);

/* emit_tr_fragment(b, smem, lay, base_elems, idx, kk, dtype, sub_elems) */
rocke_value_t* rocke_gemm_mn_emit_tr_fragment(rocke_ir_builder_t* b,
                                              rocke_value_t* smem,
                                              const rocke_gemm_mn_layout_t* lay,
                                              rocke_value_t* base_elems,
                                              int idx,
                                              int kk,
                                              const rocke_type_t* dtype,
                                              int sub_elems);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GEMM_MN_OPERAND_H */
