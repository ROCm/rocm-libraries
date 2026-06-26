// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_wmma_attention_common.c -- C99 port of
 * rocke/instances/gfx1250/_wmma_attention_common.py.
 *
 * Shared building blocks for the gfx1250 wave32 WMMA attention kernels
 * (attention_tiled_2d / attention_tiled_3d). Each helper reproduces its Python
 * counterpart's rocke_b_* builder-call sequence byte-faithfully; the emitted IR
 * is byte-identical to the Python lowerer. Args that nest two side-effecting
 * sub-calls are sequenced into temporaries to defeat C's unspecified arg-eval
 * order; const_i32 is never deduped.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_wmma_attention_common.h"

#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.spec.h"

/* _wmma_spacing(b, spacing): optional v_nop padding (spacing==0 emits nothing). */
static void rocke_g1250_wmma_spacing(rocke_ir_builder_t* b, int spacing)
{
    if(spacing > 0)
    {
        char nops[256];
        int p = 0;
        int i;
        for(i = 0; i < spacing && p < (int)sizeof(nops) - 6; ++i)
        {
            p += snprintf(nops + p, sizeof(nops) - (size_t)p, "%sv_nop", i == 0 ? "" : "\n");
        }
        (void)rocke_b_inline_asm(b, nops, "", NULL, 0, NULL, 0, NULL);
    }
}

rocke_value_t* rocke_g1250_kv_offset(rocke_ir_builder_t* b,
                                     const rocke_kv_desc_t* d,
                                     rocke_value_t* physical_block,
                                     rocke_value_t* token_in_block,
                                     rocke_value_t* kv_head,
                                     rocke_value_t* dim)
{
    /* off = mul(physical_block, const(stride_0))
     * off = add(off, mul(token_in_block, const(stride_1)))
     * off = add(off, mul(kv_head, const(stride_2)))
     * off = add(off, mul(dim, const(stride_3)))
     * Each mul has a prebound arg1 + inline const arg2 (one side-effecting), and
     * each add has prebound off arg1 -> direct nesting is order-safe. */
    rocke_value_t* off = rocke_b_mul(b, physical_block, rocke_b_const_i32(b, d->stride_0));
    off = rocke_b_add(b, off, rocke_b_mul(b, token_in_block, rocke_b_const_i32(b, d->stride_1)));
    off = rocke_b_add(b, off, rocke_b_mul(b, kv_head, rocke_b_const_i32(b, d->stride_2)));
    off = rocke_b_add(b, off, rocke_b_mul(b, dim, rocke_b_const_i32(b, d->stride_3)));
    return off;
}

const rocke_type_t* rocke_g1250_kv_storage_ir(const char* kv_storage_dtype)
{
    if(kv_storage_dtype == NULL || strcmp(kv_storage_dtype, "bf16") == 0)
    {
        return rocke_bf16();
    }
    if(strcmp(kv_storage_dtype, "fp8e4m3") == 0)
    {
        return rocke_fp8e4m3();
    }
    return NULL;
}

bool rocke_g1250_check_wmma_arch(const char* arch, char* reason, size_t reason_cap)
{
    const rocke_arch_target_t* target;
    const rocke_mma_op_t* op;
    char buf[ROCKE_ERR_MSG_CAP];

    if(arch == NULL || strcmp(arch, "gfx1250") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 WMMA attention only supports arch='gfx1250' (got '%s')",
                 arch ? arch : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target '%s'", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    op = rocke_archtarget_by_op_id(target, ROCKE_G1250_WMMA_OP_ID);
    if(op == NULL || op->family == NULL || strcmp(op->family, "wmma") != 0
       || op->wave_size != ROCKE_G1250_WAVE)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 WMMA attention requires %s wave32 WMMA",
                 ROCKE_G1250_WMMA_OP_ID);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "supported");
    return true;
}

const rocke_mma_op_t* rocke_g1250_resolve_wmma(rocke_ir_builder_t* b,
                                               const char* arch,
                                               const rocke_layout_map_t** a_layout_out,
                                               const rocke_layout_map_t** c_layout_out,
                                               int* a_frag_out,
                                               int* c_frag_out)
{
    const rocke_arch_target_t* target = rocke_archtarget_from_gfx(arch);
    const rocke_mma_op_t* op;
    if(target == NULL)
    {
        return NULL;
    }
    op = rocke_archtarget_by_op_id(target, ROCKE_G1250_WMMA_OP_ID);
    if(op == NULL)
    {
        return NULL;
    }
    if(a_layout_out)
    {
        *a_layout_out = rocke_mma_op_a_layout(op, b);
    }
    if(c_layout_out)
    {
        *c_layout_out = rocke_mma_op_c_layout(op, b);
    }
    if(a_frag_out)
    {
        *a_frag_out = op->a_frag_len;
    }
    if(c_frag_out)
    {
        *c_frag_out = op->c_frag_len;
    }
    return op;
}

/* coord helper: returns the k-component (index [1]) of a_map.coord(lane, slot). */
static rocke_value_t* rocke_g1250_coord1(rocke_ir_builder_t* b,
                                         const rocke_layout_map_t* m,
                                         rocke_value_t* lane,
                                         int slot)
{
    rocke_value_t* c0 = NULL;
    rocke_value_t* c1 = NULL;
    rocke_layout_map_coord(m, b, lane, slot, &c0, &c1);
    return c1;
}

rocke_value_t* rocke_g1250_load_kv16(rocke_ir_builder_t* b,
                                     rocke_value_t* ptr,
                                     rocke_value_t* base,
                                     rocke_value_t* scale,
                                     const rocke_type_t* kv_dtype,
                                     const rocke_type_t* out_dtype)
{
    if(kv_dtype == rocke_bf16())
    {
        return rocke_b_global_load_vN(b, ptr, base, rocke_bf16(), 16, /*align=*/16);
    }
    /* lo = load fp8x8 @base; hi = load fp8x8 @(base+8); concat(dequant lo, dequant hi) */
    {
        rocke_value_t* lo = rocke_b_global_load_vN(b, ptr, base, rocke_fp8e4m3(), 8, /*align=*/8);
        rocke_value_t* hi = rocke_b_global_load_vN(
            b, ptr, rocke_b_add(b, base, rocke_b_const_i32(b, 8)), rocke_fp8e4m3(), 8, /*align=*/8);
        rocke_value_t* dlo = rocke_dequant_fp8x8_to_dtype(b, lo, scale, out_dtype);
        rocke_value_t* dhi = rocke_dequant_fp8x8_to_dtype(b, hi, scale, out_dtype);
        return rocke_b_vec_concat(b, dlo, dhi);
    }
}

void rocke_g1250_load_q_frags(rocke_ir_builder_t* b,
                              rocke_value_t* query,
                              rocke_value_t* q_addr_row_base,
                              rocke_value_t* half_k,
                              rocke_value_t* q_valid,
                              int head_size,
                              int a_frag,
                              const rocke_type_t* dtype,
                              rocke_value_t** out_frags)
{
    /* splat = vector_splat(q_valid, a_frag); for d: select(splat, q_raw, zero). */
    rocke_value_t* splat = rocke_b_vector_splat(b, q_valid, a_frag);
    int d;
    for(d = 0; d < head_size / ROCKE_G1250_WMMA_K; ++d)
    {
        rocke_value_t* q_addr = rocke_b_add(
            b,
            rocke_b_add(b, q_addr_row_base, rocke_b_const_i32(b, d * ROCKE_G1250_WMMA_K)),
            half_k);
        rocke_value_t* q_raw = rocke_b_global_load_vN(b, query, q_addr, dtype, a_frag, a_frag * 2);
        out_frags[d] = rocke_b_vector_select(b, splat, q_raw, rocke_b_zero_vec(b, dtype, a_frag));
    }
}

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
                                   rocke_value_t* out_scores[2])
{
    int nsub;
    int d;
    for(nsub = 0; nsub < 2; ++nsub)
    {
        rocke_value_t* score = rocke_b_zero_vec_f32(b, c_frag);
        /* k_pos = add(add(tile_base, const(nsub*WMMA_N)), lane_row) */
        rocke_value_t* k_pos = rocke_b_add(
            b,
            rocke_b_add(b, tile_base, rocke_b_const_i32(b, nsub * ROCKE_G1250_WMMA_N)),
            lane_row);
        rocke_value_t* pblk = phys_block(b, k_pos, phys_ctx);
        rocke_value_t* token_in_block = rocke_b_mod(b, k_pos, rocke_b_const_i32(b, block_size));
        for(d = 0; d < head_size / ROCKE_G1250_WMMA_K; ++d)
        {
            /* dim = add(const(d*WMMA_K), half_k) */
            rocke_value_t* dim
                = rocke_b_add(b, rocke_b_const_i32(b, d * ROCKE_G1250_WMMA_K), half_k);
            rocke_value_t* k_addr
                = rocke_g1250_kv_offset(b, kv_desc, pblk, token_in_block, kv_head_idx, dim);
            rocke_value_t* k_frag = rocke_g1250_load_kv16(b, key, k_addr, k_scale, kv_dtype, dtype);
            score = rocke_b_wmma_gfx1250_f32_16x16x32_bf16(b, q_frags[d], k_frag, score);
            rocke_g1250_wmma_spacing(b, spacing);
        }
        out_scores[nsub] = score;
    }
}

/* wave_reduce_max/sum with use_dpp branch. DPP path mirrors Python
 * wave_reduce_{max,sum}(use_dpp=True): stages = log2(lanes_per_row) fused
 * vop2_f32_dpp_xor steps (masks 1<<k). lanes_per_row=16 -> 4 stages. */
static rocke_value_t* rocke_g1250_rmax(rocke_ir_builder_t* b, rocke_value_t* v, bool use_dpp)
{
    if(!use_dpp)
    {
        return rocke_wave_reduce_max(b, v, ROCKE_G1250_WAVE, 16);
    }
    {
        rocke_value_t* cur = v;
        int k;
        for(k = 0; k < 4; ++k)
        {
            cur = rocke_b_vop2_f32_dpp_xor(b, cur, 1 << k, "v_max_f32");
        }
        return cur;
    }
}

static rocke_value_t* rocke_g1250_rsum(rocke_ir_builder_t* b, rocke_value_t* v, bool use_dpp)
{
    if(!use_dpp)
    {
        return rocke_wave_reduce_sum(b, v, ROCKE_G1250_WAVE, 16);
    }
    {
        rocke_value_t* cur = v;
        int k;
        for(k = 0; k < 4; ++k)
        {
            cur = rocke_b_vop2_f32_dpp_xor(b, cur, 1 << k, "v_add_f32");
        }
        return cur;
    }
}

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
                                    rocke_value_t* p_out[2])
{
    rocke_value_t* rm0;
    rocke_value_t* rm1;
    rocke_value_t* has0;
    rocke_value_t* has1;
    rocke_value_t* tile_has;
    rocke_value_t* m_cand;
    rocke_value_t* m_new;
    rocke_value_t* alpha;
    rocke_value_t* p0;
    rocke_value_t* p1;
    rocke_value_t* rs0;
    rocke_value_t* rs1;
    rocke_value_t* l_new;

    rm0 = rocke_g1250_rmax(b, srs[0], use_dpp);
    rm1 = rocke_g1250_rmax(b, srs[1], use_dpp);
    has0 = rocke_b_fcmp(b, "ogt", rm0, neg_inf);
    has1 = rocke_b_fcmp(b, "ogt", rm1, neg_inf);
    tile_has = rocke_b_lor(b, has0, has1);
    m_cand = rocke_b_fmax(b, rm0, rm1);
    /* m_new = select(tile_has, fmax(m_prev, m_cand), m_prev) */
    m_new = rocke_b_select(b, tile_has, rocke_b_fmax(b, m_prev, m_cand), m_prev);
    /* alpha = exp2(fsub(m_prev, m_new)) */
    alpha = rocke_b_exp2(b, rocke_b_fsub(b, m_prev, m_new));
    /* p0 = select(has0, exp2(fsub(srs0, m_new)), zero_f) */
    p0 = rocke_b_select(b, has0, rocke_b_exp2(b, rocke_b_fsub(b, srs[0], m_new)), zero_f);
    p1 = rocke_b_select(b, has1, rocke_b_exp2(b, rocke_b_fsub(b, srs[1], m_new)), zero_f);
    rs0 = rocke_g1250_rsum(b, p0, use_dpp);
    rs1 = rocke_g1250_rsum(b, p1, use_dpp);
    /* l_new = fadd(fmul(l_prev, alpha), fadd(rs0, rs1)). Python evals the fmul
     * (arg1) before the inner fadd (arg2); sequence to preserve SSA order. */
    {
        rocke_value_t* la = rocke_b_fmul(b, l_prev, alpha);
        rocke_value_t* rr = rocke_b_fadd(b, rs0, rs1);
        l_new = rocke_b_fadd(b, la, rr);
    }

    *m_new_out = m_new;
    *l_new_out = l_new;
    *alpha_out = alpha;
    p_out[0] = p0;
    p_out[1] = p1;
}

/* Shared V-tile staging (token-major or buffered). buf_idx NULL => [lane, dim]. */
static void rocke_g1250_stage_v_common(rocke_ir_builder_t* b,
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
                                       void* phys_ctx)
{
    rocke_value_t* v_global = rocke_b_add(b, tile_base, lane);
    rocke_value_t* vpblk = phys_block(b, v_global, phys_ctx);
    rocke_value_t* v_tib = rocke_b_mod(b, v_global, rocke_b_const_i32(b, block_size));
    rocke_value_t* v_row_base
        = rocke_g1250_kv_offset(b, kv_desc, vpblk, v_tib, kv_head_idx, rocke_b_const_i32(b, 0));
    int dd;
    for(dd = 0; dd < head_size / 8; ++dd)
    {
        rocke_value_t* v8;
        rocke_value_t* off = rocke_b_add(b, v_row_base, rocke_b_const_i32(b, dd * 8));
        if(kv_dtype == rocke_bf16())
        {
            v8 = rocke_b_global_load_vN(b, value, off, rocke_bf16(), 8, /*align=*/16);
        }
        else
        {
            rocke_value_t* raw
                = rocke_b_global_load_vN(b, value, off, rocke_fp8e4m3(), 8, /*align=*/8);
            v8 = rocke_dequant_fp8x8_to_dtype(b, raw, v_scale, dtype);
        }
        {
            rocke_value_t* idx[3];
            if(buf_idx == NULL)
            {
                idx[0] = lane;
                idx[1] = rocke_b_const_i32(b, dd * 8);
                rocke_b_smem_store_vN(b, V_lds, idx, 2, v8, 8);
            }
            else
            {
                idx[0] = buf_idx;
                idx[1] = lane;
                idx[2] = rocke_b_const_i32(b, dd * 8);
                rocke_b_smem_store_vN(b, V_lds, idx, 3, v8, 8);
            }
        }
    }
}

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
                              void* phys_ctx)
{
    rocke_g1250_stage_v_common(b,
                               V_lds,
                               NULL,
                               value,
                               kv_desc,
                               kv_head_idx,
                               tile_base,
                               lane,
                               block_size,
                               head_size,
                               kv_dtype,
                               v_scale,
                               dtype,
                               phys_block,
                               phys_ctx);
}

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
                                  void* phys_ctx)
{
    rocke_g1250_stage_v_common(b,
                               V_lds,
                               buf_idx,
                               value,
                               kv_desc,
                               kv_head_idx,
                               tile_base,
                               lane,
                               block_size,
                               head_size,
                               kv_dtype,
                               v_scale,
                               dtype,
                               phys_block,
                               phys_ctx);
}

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
                                         rocke_value_t* buf_idx)
{
    rocke_value_t* v_global = rocke_b_add(b, tile_base, lane);
    rocke_value_t* vpblk = phys_block(b, v_global, phys_ctx);
    rocke_value_t* v_tib = rocke_b_mod(b, v_global, rocke_b_const_i32(b, block_size));
    rocke_value_t* v_row_base
        = rocke_g1250_kv_offset(b, kv_desc, vpblk, v_tib, kv_head_idx, rocke_b_const_i32(b, 0));
    int dd;
    int i;
    for(dd = 0; dd < head_size / 8; ++dd)
    {
        rocke_value_t* v8;
        rocke_value_t* off = rocke_b_add(b, v_row_base, rocke_b_const_i32(b, dd * 8));
        if(kv_dtype == rocke_bf16())
        {
            v8 = rocke_b_global_load_vN(b, value, off, rocke_bf16(), 8, /*align=*/16);
        }
        else
        {
            rocke_value_t* raw
                = rocke_b_global_load_vN(b, value, off, rocke_fp8e4m3(), 8, /*align=*/8);
            v8 = rocke_dequant_fp8x8_to_dtype(b, raw, v_scale, dtype);
        }
        for(i = 0; i < 8; ++i)
        {
            rocke_value_t* row = rocke_b_const_i32(b, dd * 8 + i);
            rocke_value_t* idx[3];
            rocke_value_t* elem = rocke_b_vec_extract(b, v8, i);
            if(buf_idx == NULL)
            {
                idx[0] = row;
                idx[1] = lane;
                rocke_b_smem_store_vN(b, V_lds_T, idx, 2, elem, 1);
            }
            else
            {
                idx[0] = buf_idx;
                idx[1] = row;
                idx[2] = lane;
                rocke_b_smem_store_vN(b, V_lds_T, idx, 3, elem, 1);
            }
        }
    }
}

/* _compute_pv_inner(b, p_a, V_lds, accs, ...). Writes back into accs[]. */
static void rocke_g1250_compute_pv_inner(rocke_ir_builder_t* b,
                                         rocke_value_t* p_a,
                                         rocke_value_t* V_lds,
                                         rocke_value_t** accs,
                                         const rocke_layout_map_t* a_map,
                                         rocke_value_t* lane,
                                         rocke_value_t* col,
                                         int a_frag,
                                         int head_size,
                                         const rocke_type_t* dtype,
                                         rocke_value_t* v_extra_idx,
                                         int spacing)
{
    int d;
    int j;
    for(d = 0; d < head_size / ROCKE_G1250_WMMA_N; ++d)
    {
        rocke_value_t* d_col = rocke_b_add(b, rocke_b_const_i32(b, d * ROCKE_G1250_WMMA_N), col);
        rocke_value_t* v_b = rocke_b_zero_vec(b, dtype, a_frag);
        for(j = 0; j < a_frag; ++j)
        {
            rocke_value_t* v_k = rocke_g1250_coord1(b, a_map, lane, j);
            rocke_value_t* idx[3];
            rocke_value_t* loaded;
            if(v_extra_idx == NULL)
            {
                idx[0] = v_k;
                idx[1] = d_col;
                loaded = rocke_b_smem_load_vN(b, V_lds, idx, 2, dtype, 1);
            }
            else
            {
                idx[0] = v_extra_idx;
                idx[1] = v_k;
                idx[2] = d_col;
                loaded = rocke_b_smem_load_vN(b, V_lds, idx, 3, dtype, 1);
            }
            v_b = rocke_b_vec_insert(b, v_b, rocke_b_vec_extract(b, loaded, 0), j);
        }
        accs[d] = rocke_b_wmma_gfx1250_f32_16x16x32_bf16(b, p_a, v_b, accs[d]);
        rocke_g1250_wmma_spacing(b, spacing);
    }
}

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
                            int spacing)
{
    rocke_value_t* p_a = rocke_b_zero_vec(b, dtype, a_frag);
    int j;
    (void)c_frag;
    for(j = 0; j < a_frag; ++j)
    {
        rocke_value_t* a_k = rocke_g1250_coord1(b, a_map, lane, j);
        rocke_value_t* idx[3];
        rocke_value_t* p_load;
        if(p_extra_idx == NULL)
        {
            idx[0] = lane_row;
            idx[1] = a_k;
            p_load = rocke_b_smem_load_vN(b, P_lds, idx, 2, dtype, 1);
        }
        else
        {
            idx[0] = p_extra_idx;
            idx[1] = lane_row;
            idx[2] = a_k;
            p_load = rocke_b_smem_load_vN(b, P_lds, idx, 3, dtype, 1);
        }
        p_a = rocke_b_vec_insert(b, p_a, rocke_b_vec_extract(b, p_load, 0), j);
    }
    rocke_g1250_compute_pv_inner(
        b, p_a, V_lds, accs, a_map, lane, col, a_frag, head_size, dtype, v_extra_idx, spacing);
}

/* _wide_frag(smem, row, extra): read a_frag contiguous tokens at k0 as ceil(a_frag/8)
 * wide chunks (vec<=8) and concat. */
static rocke_value_t* rocke_g1250_wide_frag(rocke_ir_builder_t* b,
                                            rocke_value_t* smem,
                                            rocke_value_t* row,
                                            rocke_value_t* extra,
                                            rocke_value_t* k0,
                                            int a_frag,
                                            const rocke_type_t* dtype)
{
    const int CH = 8;
    rocke_value_t* frag = NULL;
    int off;
    for(off = 0; off < a_frag; off += CH)
    {
        int n = (CH < a_frag - off) ? CH : (a_frag - off);
        rocke_value_t* kk = rocke_b_add(b, k0, rocke_b_const_i32(b, off));
        rocke_value_t* idx[3];
        rocke_value_t* chunk;
        if(extra == NULL)
        {
            idx[0] = row;
            idx[1] = kk;
            chunk = rocke_b_smem_load_vN(b, smem, idx, 2, dtype, n);
        }
        else
        {
            idx[0] = extra;
            idx[1] = row;
            idx[2] = kk;
            chunk = rocke_b_smem_load_vN(b, smem, idx, 3, dtype, n);
        }
        frag = (frag == NULL) ? chunk : rocke_b_vec_concat(b, frag, chunk);
    }
    return frag;
}

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
                                 int spacing)
{
    /* k0 = mul(div(lane, const16), const16); col = mod(lane, const16).
     * Sequence the div (arg1) before the trailing const(16) of the mul. */
    rocke_value_t* k0_div = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
    rocke_value_t* k0 = rocke_b_mul(b, k0_div, rocke_b_const_i32(b, 16));
    rocke_value_t* col = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));
    rocke_value_t* p_a;
    int d;
    (void)a_map;
    p_a = rocke_g1250_wide_frag(b, P_lds, lane_row, p_extra_idx, k0, a_frag, dtype);
    for(d = 0; d < head_size / ROCKE_G1250_WMMA_N; ++d)
    {
        rocke_value_t* d_col = rocke_b_add(b, rocke_b_const_i32(b, d * ROCKE_G1250_WMMA_N), col);
        rocke_value_t* v_b
            = rocke_g1250_wide_frag(b, V_lds_T, d_col, v_extra_idx, k0, a_frag, dtype);
        accs[d] = rocke_b_wmma_gfx1250_f32_16x16x32_bf16(b, p_a, v_b, accs[d]);
        rocke_g1250_wmma_spacing(b, spacing);
    }
}

/* _bpermute_vec8(b, vec8, src_lane, dtype): ds_bpermute a <8 x dtype> (16-bit). */
static rocke_value_t* rocke_g1250_bpermute_vec8(rocke_ir_builder_t* b,
                                                rocke_value_t* vec8,
                                                rocke_value_t* src_lane,
                                                const rocke_type_t* dtype)
{
    rocke_value_t* src_addr = rocke_b_mul(b, src_lane, rocke_b_const_i32(b, 4));
    rocke_value_t* i32v = rocke_b_bitcast(b, vec8, rocke_vector_type(b, rocke_i32(), 4));
    rocke_value_t* out = rocke_b_zero_vec(b, rocke_i32(), 4);
    int j;
    for(j = 0; j < 4; ++j)
    {
        out = rocke_b_vec_insert(
            b, out, rocke_b_ds_bpermute(b, src_addr, rocke_b_vec_extract(b, i32v, j)), j);
    }
    return rocke_b_bitcast(b, out, rocke_vector_type(b, dtype, 8));
}

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
                                 int spacing)
{
    rocke_value_t* p_a = rocke_b_zero_vec(b, dtype, a_frag);
    rocke_value_t* lt8;
    rocke_value_t* partner;
    rocke_value_t* splat16;
    int j;
    int d;
    /* A (P) operand: scalar a_map gather (same as compute_pv). */
    for(j = 0; j < a_frag; ++j)
    {
        rocke_value_t* a_k = rocke_g1250_coord1(b, a_map, lane, j);
        rocke_value_t* idx[3];
        rocke_value_t* p_load;
        if(p_extra_idx == NULL)
        {
            idx[0] = lane_row;
            idx[1] = a_k;
            p_load = rocke_b_smem_load_vN(b, P_lds, idx, 2, dtype, 1);
        }
        else
        {
            idx[0] = p_extra_idx;
            idx[1] = lane_row;
            idx[2] = a_k;
            p_load = rocke_b_smem_load_vN(b, P_lds, idx, 3, dtype, 1);
        }
        p_a = rocke_b_vec_insert(b, p_a, rocke_b_vec_extract(b, p_load, 0), j);
    }

    /* lt8 = cmp_lt(mod(lane, const16), const8); partner = xor(lane, const8) */
    lt8 = rocke_b_cmp_lt(
        b, rocke_b_mod(b, lane, rocke_b_const_i32(b, 16)), rocke_b_const_i32(b, 8));
    partner = rocke_b_xor(b, lane, rocke_b_const_i32(b, 8));
    splat16 = rocke_b_vector_splat(b, lt8, 16);
    for(d = 0; d < head_size / ROCKE_G1250_WMMA_N; ++d)
    {
        int c0 = d * ROCKE_G1250_WMMA_N;
        rocke_value_t* r0;
        rocke_value_t* r1;
        rocke_value_t* p0;
        rocke_value_t* p1;
        rocke_value_t* v_b;
        if(v_extra_idx == NULL)
        {
            rocke_value_t* i0[2];
            rocke_value_t* i1[2];
            i0[0] = lane;
            i0[1] = rocke_b_const_i32(b, c0);
            r0 = rocke_b_ds_read_tr16_b128(b, V_lds, i0, 2, dtype);
            i1[0] = lane;
            i1[1] = rocke_b_const_i32(b, c0 + 8);
            r1 = rocke_b_ds_read_tr16_b128(b, V_lds, i1, 2, dtype);
        }
        else
        {
            rocke_value_t* i0[3];
            rocke_value_t* i1[3];
            i0[0] = v_extra_idx;
            i0[1] = lane;
            i0[2] = rocke_b_const_i32(b, c0);
            r0 = rocke_b_ds_read_tr16_b128(b, V_lds, i0, 3, dtype);
            i1[0] = v_extra_idx;
            i1[1] = lane;
            i1[2] = rocke_b_const_i32(b, c0 + 8);
            r1 = rocke_b_ds_read_tr16_b128(b, V_lds, i1, 3, dtype);
        }
        p0 = rocke_g1250_bpermute_vec8(b, r0, partner, dtype);
        p1 = rocke_g1250_bpermute_vec8(b, r1, partner, dtype);
        /* v_b = vector_select(splat16, vec_concat(r0, p0), vec_concat(p1, r1)) */
        v_b = rocke_b_vector_select(
            b, splat16, rocke_b_vec_concat(b, r0, p0), rocke_b_vec_concat(b, p1, r1));
        accs[d] = rocke_b_wmma_gfx1250_f32_16x16x32_bf16(b, p_a, v_b, accs[d]);
        rocke_g1250_wmma_spacing(b, spacing);
    }
}

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
                                       int spacing)
{
    rocke_value_t* p_a = rocke_b_zero_vec(b, dtype, a_frag);
    rocke_value_t* lane_half = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
    rocke_value_t* lane_col = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));
    rocke_value_t* row_half = rocke_b_div(b, lane_col, rocke_b_const_i32(b, 8));
    rocke_value_t* row_reg = rocke_b_mod(b, lane_col, rocke_b_const_i32(b, 8));
    int j;
    int r;
    (void)c_frag;
    for(j = 0; j < a_frag; ++j)
    {
        /* src_lane = add(mul(row_half, const16), const(j)); src_addr = mul(src_lane, const4) */
        rocke_value_t* src_lane = rocke_b_add(
            b, rocke_b_mul(b, row_half, rocke_b_const_i32(b, 16)), rocke_b_const_i32(b, j));
        rocke_value_t* src_addr = rocke_b_mul(b, src_lane, rocke_b_const_i32(b, 4));
        rocke_value_t* elt = rocke_b_const_f32(b, 0.0);
        for(r = 0; r < 8; ++r)
        {
            rocke_value_t* p0_i = rocke_b_bitcast(b, ps0[r], rocke_i32());
            rocke_value_t* p1_i = rocke_b_bitcast(b, ps1[r], rocke_i32());
            rocke_value_t* from_p0
                = rocke_b_bitcast(b, rocke_b_ds_bpermute(b, src_addr, p0_i), rocke_f32());
            rocke_value_t* from_p1
                = rocke_b_bitcast(b, rocke_b_ds_bpermute(b, src_addr, p1_i), rocke_f32());
            /* from_col_half = select(cmp_eq(lane_half, 0), from_p0, from_p1) */
            rocke_value_t* from_col_half = rocke_b_select(
                b, rocke_b_cmp_eq(b, lane_half, rocke_b_const_i32(b, 0)), from_p0, from_p1);
            /* elt = select(cmp_eq(row_reg, r), from_col_half, elt) */
            elt = rocke_b_select(
                b, rocke_b_cmp_eq(b, row_reg, rocke_b_const_i32(b, r)), from_col_half, elt);
        }
        p_a = rocke_b_vec_insert(b, p_a, rocke_b_cast_f32_to(b, elt, dtype), j);
    }
    rocke_g1250_compute_pv_inner(
        b, p_a, V_lds, accs, a_map, lane, col, a_frag, head_size, dtype, v_extra_idx, spacing);
}
