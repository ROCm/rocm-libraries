// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_build_8c.c -- C99 port of build_direct_conv_8c
 * (rocke/instances/common/conv_direct_grouped.py, lines 1215-1555).
 *
 * The 8c kernel uses mfma_f32_16x16x16_f16 (K=16) with two filter S-positions
 * (s=0 and s=1) folded into the K=16 dimension.  The residual s=2 column is
 * handled as a zero-padded K=16 atom (lanes c4 in {2,3} carry zeros).
 * Structurally identical to the 16c non-fold_k32 path except:
 *   - chunk_desc has ch_block_dim=2 (cpg=8 → 2 blocks of 4 channels per group).
 *   - weights_main[r]: one folded K=16 vec4 (s_lane selects s from c4).
 *   - weights_s2[r]:   one s=2 residual K=16 vec4 (zeroed for c4 in {2,3}).
 *   - Two MFMAs per (qt, r): one for the folded main atom, one for the residual.
 *   - Store gate: c4 < kpg/4 = 2 (only c4 in {0,1} carry valid k_out rows).
 *
 * Phase functions: prologue, load_weights, build_chunk_meta, build_descriptors,
 * prologue_prefetch, stream_h_loop.  Byte-identical to the Python source.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  Prologue  (Python lines 1239-1319)
 * ===================================================================== */
bool rocke_dconv8c_prologue(rocke_dconv_8c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_8c_spec_t* spec = ctx->spec;
    char reason[ROCKE_ERR_MSG_CAP];

    if(rocke_direct_conv_8c_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    if(!rocke_direct_conv_8c_is_valid_spec(spec, ctx->arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->p = spec->problem;

    /* ---- block geometry ---- */
    ctx->BLOCK_Q = spec->block_q;
    ctx->BLOCK_GROUPS = spec->block_groups;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_conv_8c_threads_per_block(spec);
    ctx->LDS_W = ctx->BLOCK_Q + ctx->p.KW - 1;
    ctx->LDS_ROW_FP16 = ctx->LDS_W * ctx->BLOCK_GROUPS * ctx->p.cpg;
    ctx->LOAD_VEC = 4;
    ctx->NUM_VEC4 = ctx->LDS_ROW_FP16 / ctx->LOAD_VEC;
    if(ctx->NUM_VEC4 == 0)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    ctx->PASSES = (ctx->NUM_VEC4 + ctx->THREADS - 1) / ctx->THREADS;
    ctx->q_subtiles = ctx->BLOCK_Q / 16;
    ctx->n_iters = ctx->p.H + ctx->p.KH - 1;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

    /* ---- params ---- */
    {
        const rocke_type_t* f16ptr = rocke_ptr_type(b, rocke_f16(), "global");
        rocke_param_opts_t ro;
        rocke_param_opts_t wo;
        rocke_param_opts_t none;

        ro = (rocke_param_opts_t){0};
        ro.noalias = true;
        ro.noalias_set = true;
        ro.readonly = true;
        ro.readonly_set = true;
        ro.align = 16;
        ro.align_set = true;
        ctx->A = rocke_b_param(b, "A", f16ptr, &ro);
        ctx->Bp = rocke_b_param(b, "B", f16ptr, &ro);

        wo = (rocke_param_opts_t){0};
        wo.noalias = true;
        wo.noalias_set = true;
        wo.writeonly = true;
        wo.writeonly_set = true;
        wo.align = 16;
        wo.align_set = true;
        ctx->D = rocke_b_param(b, "D", f16ptr, &wo);

        none = (rocke_param_opts_t){0};
        ctx->A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
        ctx->B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
        ctx->D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);
    }

    /* ---- SSA constants (Python lines 1266-1274) ---- */
    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    ctx->c_BG = rocke_b_const_i32(b, ctx->BLOCK_GROUPS);
    ctx->c_BQ = rocke_b_const_i32(b, ctx->BLOCK_Q);
    ctx->c_cpg = rocke_b_const_i32(b, ctx->p.cpg);
    ctx->c_kpg = rocke_b_const_i32(b, ctx->p.kpg);
    ctx->c_W = rocke_b_const_i32(b, ctx->p.W);
    ctx->c_BG_cpg = rocke_b_const_i32(b, ctx->BLOCK_GROUPS * ctx->p.cpg);
    ctx->c_half_bytes = rocke_b_const_i32(b, 2);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);

    /* ---- thread / wave / lane decode (Python lines 1277-1290) ---- */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);
    ctx->c4 = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, 16));
    ctx->q_in_lane = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, 16));

    /* s_lane = c4 / 2  (Python: b.div(c4, b.const_i32(2)))
     * ch_lane = (c4 % 2) * 4  (Python: b.mul(b.mod(c4, b.const_i32(2)), b.const_i32(4)))
     * Force Python left-to-right SSA emission. */
    ctx->s_lane = rocke_b_div(b, ctx->c4, rocke_b_const_i32(b, 2));
    {
        rocke_value_t* mod_c4_2 = rocke_b_mod(b, ctx->c4, rocke_b_const_i32(b, 2));
        ctx->ch_lane = rocke_b_mul(b, mod_c4_2, rocke_b_const_i32(b, 4));
    }
    /* lane_in_lo_half = b.cmp_lt(c4, b.const_i32(2)) */
    ctx->lane_in_lo_half = rocke_b_cmp_lt(b, ctx->c4, rocke_b_const_i32(b, 2));

    /* ---- grid / group decode (Python lines 1297-1302) ---- */
    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->n = rocke_b_block_id_z(b);
    ctx->g_tile = ctx->by;
    ctx->g = rocke_b_add(b, rocke_b_mul(b, ctx->g_tile, ctx->c_BG), ctx->wave_id);
    ctx->q_tile_start = rocke_b_mul(b, ctx->bx, ctx->c_BQ);

    /* ---- LDS (Python lines 1305-1315) ---- */
    ctx->lds_total_fp16 = ctx->PASSES * ctx->THREADS * ctx->LOAD_VEC;
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = ctx->lds_total_fp16;
        ctx->A_smem = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "lds_a");
        if(spec->double_buffer)
        {
            ctx->B_smem = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "lds_b");
        }
        else
        {
            ctx->B_smem = ctx->A_smem;
        }
    }

    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);
    ctx->d_rsrc = rocke_b_buffer_rsrc(b, ctx->D, ctx->D_bytes);

    ctx->fp16x4_zero = rocke_b_zero_vec_f16(b, 4);
    ctx->zero_acc = rocke_b_zero_vec_f32(b, 4);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Weight-load phase  (Python lines 1320-1353)
 *
 *  For each r_const:
 *    weights_main[r]: folded K=16 vec4 — s_lane selects s=0 (c4∈{0,1}) or
 *                     s=1 (c4∈{2,3}), ch_lane selects channel block.
 *    weights_s2[r]:   s=2 residual — valid only for c4∈{0,1}; zeroed otherwise.
 * ===================================================================== */
void rocke_dconv8c_load_weights(rocke_dconv_8c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);
    int r_const;

    /* b_desc = TensorDescriptor.naive("B", [total_k, KH, KW, cpg],
     *                                 coord_names=("k_out","r","s","c")) */
    {
        int lengths[4];
        static const char* const coord_names[4] = {"k_out", "r", "s", "c"};
        lengths[0] = total_k;
        lengths[1] = ctx->p.KH;
        lengths[2] = ctx->p.KW;
        lengths[3] = ctx->p.cpg;
        ctx->b_desc = rocke_tensor_descriptor_naive(b, "B", lengths, 4, NULL, coord_names, 4);
    }

    /* k_out_val = g*kpg + q_in_lane */
    ctx->k_out_val = rocke_b_add(b, rocke_b_mul(b, ctx->g, ctx->c_kpg), ctx->q_in_lane);

    ctx->n_weights = 0;

    for(r_const = 0; r_const < ctx->p.KH; ++r_const)
    {
        rocke_value_t* r_i = rocke_b_const_i32(b, r_const);
        rocke_value_t* w_off_main = NULL;
        rocke_value_t* w_off_s2 = NULL;
        rocke_value_t* valid = NULL;
        rocke_value_t* w_s2 = NULL;

        /* Main atom: s_lane selects s=0 (c4∈{0,1}) or s=1 (c4∈{2,3}).
         * w_off_main, _ = b_desc.offset(b, k_out=k_out_val, r=r_i, s=s_lane, c=ch_lane) */
        {
            const char* in_names[4] = {"k_out", "r", "s", "c"};
            rocke_value_t* in_values[4];
            in_values[0] = ctx->k_out_val;
            in_values[1] = r_i;
            in_values[2] = ctx->s_lane;
            in_values[3] = ctx->ch_lane;
            rocke_transforms_descriptor_offset(
                b, ctx->b_desc, in_names, in_values, 4, &w_off_main, &valid);
        }
        ctx->weights_main[r_const] = rocke_b_buffer_load_vN_f16(
            b, ctx->b_rsrc, rocke_b_mul(b, w_off_main, ctx->c_half_bytes), ctx->c0, 2);

        /* Residual s=2 atom: valid only for c4∈{0,1}; zeroed for c4∈{2,3}. */
        {
            const char* in_names[4] = {"k_out", "r", "s", "c"};
            rocke_value_t* in_values[4];
            in_values[0] = ctx->k_out_val;
            in_values[1] = r_i;
            in_values[2] = rocke_b_const_i32(b, 2);
            in_values[3] = ctx->ch_lane;
            rocke_transforms_descriptor_offset(
                b, ctx->b_desc, in_names, in_values, 4, &w_off_s2, &valid);
        }
        w_s2 = rocke_b_buffer_load_vN_f16(
            b, ctx->b_rsrc, rocke_b_mul(b, w_off_s2, ctx->c_half_bytes), ctx->c0, 2);
        ctx->weights_s2[r_const] = rocke_b_select(b, ctx->lane_in_lo_half, w_s2, ctx->fp16x4_zero);
    }
    ctx->n_weights = ctx->p.KH;
}

/* ===================================================================== *
 *  Chunk-decode phase  (Python lines 1358-1387)
 *
 *  chunk_desc = naive("chunk_unmerge", [LDS_W, BLOCK_GROUPS, 2],
 *                     coord_names=(W_lds, group_in_wg, ch_block))
 *               .transform(unmerge_magic(...))
 *  Note: ch_block_dim = 2 (cpg=8 → 2 blocks of 4 channels, not 4 as in 16c).
 * ===================================================================== */
void rocke_dconv8c_build_chunk_meta(rocke_dconv_8c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    static const char* const coord_names[3] = {"W_lds", "group_in_wg", "ch_block"};
    int lengths[3];
    int pass_idx;
    rocke_tensor_descriptor_t* naive;

    lengths[0] = ctx->LDS_W;
    lengths[1] = ctx->BLOCK_GROUPS;
    lengths[2] = 2; /* cpg=8: 2 blocks of 4 channels */

    naive = rocke_tensor_descriptor_naive(b, "chunk_unmerge", lengths, 3, NULL, coord_names, 3);

    {
        const rocke_transform_t* xforms[1];
        xforms[0] = rocke_unmerge_magic(b, "chunk_idx", coord_names, 3, lengths);
        ctx->chunk_desc = rocke_tensor_descriptor_transform(b, naive, xforms, 1);
    }

    ctx->n_chunk_meta = 0;
    for(pass_idx = 0; pass_idx < ctx->PASSES; ++pass_idx)
    {
        rocke_value_t* chunk_idx
            = rocke_b_add(b, ctx->tid, rocke_b_const_i32(b, pass_idx * ctx->THREADS));

        const char* out_names[8];
        rocke_value_t* out_values[8];
        const char* in_names[1] = {"chunk_idx"};
        rocke_value_t* in_values[1];
        rocke_value_t* ch_block = NULL;
        rocke_value_t* group_in_wg = NULL;
        rocke_value_t* W_lds = NULL;
        int n_out, i;

        in_values[0] = chunk_idx;
        n_out
            = rocke_tensor_descriptor_unmerge_lower(b,
                                                    ctx->chunk_desc,
                                                    in_names,
                                                    in_values,
                                                    1,
                                                    out_names,
                                                    out_values,
                                                    (int)(sizeof out_names / sizeof out_names[0]));
        for(i = 0; i < n_out; ++i)
        {
            if(out_names[i] && out_names[i][0] == 'c' && out_names[i][1] == 'h'
               && out_names[i][2] == '_')
            {
                ch_block = out_values[i];
            }
            else if(out_names[i] && out_names[i][0] == 'g')
            {
                group_in_wg = out_values[i];
            }
            else if(out_names[i] && out_names[i][0] == 'W')
            {
                W_lds = out_values[i];
            }
        }

        ctx->chunk_meta[pass_idx].chunk_idx = chunk_idx;
        ctx->chunk_meta[pass_idx].ch_block = ch_block;
        ctx->chunk_meta[pass_idx].group_in_wg = group_in_wg;
        ctx->chunk_meta[pass_idx].W_lds = W_lds;
        ctx->chunk_meta[pass_idx].in_bounds
            = rocke_b_cmp_lt(b, chunk_idx, rocke_b_const_i32(b, ctx->NUM_VEC4));
        ctx->chunk_meta[pass_idx].abs_group
            = rocke_b_add(b, rocke_b_mul(b, ctx->g_tile, ctx->c_BG), group_in_wg);
        ctx->n_chunk_meta++;
    }
}

/* ===================================================================== *
 *  Descriptor phase  (Python lines 1389-1410, 1478-1482)
 * ===================================================================== */
void rocke_dconv8c_build_descriptors(rocke_dconv_8c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int total_c = rocke_direct_conv_problem_total_c(&ctx->p);
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);

    {
        static const char* const a_coords[4] = {"n", "h", "w", "c"};
        int a_lengths[4];
        rocke_tensor_descriptor_t* a_naive;
        const rocke_transform_t* xforms[2];

        a_lengths[0] = ctx->p.N;
        a_lengths[1] = ctx->p.H;
        a_lengths[2] = ctx->p.W;
        a_lengths[3] = total_c;
        a_naive = rocke_tensor_descriptor_naive(b, "A", a_lengths, 4, NULL, a_coords, 4);

        {
            static const char* const h_upper[1] = {"y_iter"};
            int h_strides[1] = {1};
            xforms[0]
                = rocke_embed_bounded(b, h_upper, 1, "h", h_strides, -ctx->p.PAD, 0, ctx->p.H);
        }
        {
            static const char* const w_upper[2] = {"q_pos", "W_lds_pos"};
            int w_strides[2] = {1, 1};
            xforms[1]
                = rocke_embed_bounded(b, w_upper, 2, "w", w_strides, -ctx->p.PAD, 0, ctx->p.W);
        }
        ctx->a_desc = rocke_tensor_descriptor_transform(b, a_naive, xforms, 2);
    }

    {
        static const char* const d_coords[4] = {"n", "h", "w", "k"};
        int d_lengths[4];
        d_lengths[0] = ctx->p.N;
        d_lengths[1] = ctx->p.H;
        d_lengths[2] = ctx->p.W;
        d_lengths[3] = total_k;
        ctx->d_desc = rocke_tensor_descriptor_naive(b, "D", d_lengths, 4, NULL, d_coords, 4);
    }
}

/* ===================================================================== *
 *  Helper: issue_dram_load for 8c.
 *  Identical logic to the 16c version — the chunk_meta fields have the same
 *  meaning; only ch_block_dim differs (2 instead of 4 in chunk_desc, but the
 *  load formula is the same: abs_group*cpg + ch_block*4).
 * ===================================================================== */
static int rocke_dconv8c_issue_dram_load(rocke_dconv_8c_ctx_t* ctx,
                                         rocke_value_t* y_iter_val,
                                         rocke_value_t** out_vecs,
                                         rocke_value_t** out_lds_idx,
                                         int out_cap)
{
    rocke_ir_builder_t* b = ctx->b;
    int count = 0;
    int i;

    for(i = 0; i < ctx->n_chunk_meta; ++i)
    {
        rocke_value_t* c_val;
        rocke_value_t* a_off_elems;
        rocke_value_t* addr_valid;
        rocke_value_t* valid;
        rocke_value_t* a_off_bytes;
        rocke_value_t* safe_off;
        rocke_value_t* a_vec;
        rocke_value_t* lds_idx;
        const char* off_names[5];
        rocke_value_t* off_vals[5];

        if(count >= out_cap)
        {
            return -1;
        }

        /* c_val = abs_group*cpg + ch_block*4 -- force Python left-to-right SSA. */
        {
            rocke_value_t* mul_ag = rocke_b_mul(b, ctx->chunk_meta[i].abs_group, ctx->c_cpg);
            rocke_value_t* mul_cb
                = rocke_b_mul(b, ctx->chunk_meta[i].ch_block, rocke_b_const_i32(b, 4));
            c_val = rocke_b_add(b, mul_ag, mul_cb);
        }

        off_names[0] = "n";
        off_vals[0] = ctx->n;
        off_names[1] = "y_iter";
        off_vals[1] = y_iter_val;
        off_names[2] = "q_pos";
        off_vals[2] = ctx->q_tile_start;
        off_names[3] = "W_lds_pos";
        off_vals[3] = ctx->chunk_meta[i].W_lds;
        off_names[4] = "c";
        off_vals[4] = c_val;
        if(!rocke_transforms_descriptor_offset(
               b, ctx->a_desc, off_names, off_vals, 5, &a_off_elems, &addr_valid))
        {
            return -1;
        }

        valid = rocke_b_land(b, addr_valid, ctx->chunk_meta[i].in_bounds);
        a_off_bytes = rocke_b_mul(b, a_off_elems, ctx->c_half_bytes);
        safe_off = rocke_b_select(b, valid, a_off_bytes, ctx->oob_sentinel);
        a_vec = rocke_b_buffer_load_vN_f16(b, ctx->a_rsrc, safe_off, ctx->c0, 2);
        a_vec = rocke_b_select(b, valid, a_vec, ctx->fp16x4_zero);
        lds_idx = rocke_b_mul(b, ctx->chunk_meta[i].chunk_idx, rocke_b_const_i32(b, 4));

        out_vecs[count] = a_vec;
        out_lds_idx[count] = lds_idx;
        ++count;
    }
    return count;
}

static void rocke_dconv8c_store_to_lds(rocke_dconv_8c_ctx_t* ctx,
                                       rocke_value_t* const* vecs,
                                       rocke_value_t* const* lds_idx,
                                       int n,
                                       rocke_value_t* lds)
{
    rocke_ir_builder_t* b = ctx->b;
    int i;
    for(i = 0; i < n; ++i)
    {
        rocke_value_t* indices[2];
        indices[0] = ctx->c0;
        indices[1] = lds_idx[i];
        rocke_b_smem_store_vN_f16(b, lds, indices, 2, vecs[i], 4);
    }
}

/* lds_read_input_main(q_subtile, lds) -- per-lane <4 x half> read.
 * W_lds_idx = q_in_lane + q_subtile*16 + s_lane
 * ch_block_idx = ch_lane / 4
 * lds_idx = W_lds_idx*BG_cpg + wave_id*cpg + ch_block_idx*4 */
static rocke_value_t*
    rocke_dconv8c_lds_read_main(rocke_dconv_8c_ctx_t* ctx, int q_subtile, rocke_value_t* lds)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* W_lds_idx;
    rocke_value_t* lds_idx;
    rocke_value_t* indices[2];

    /* W_lds_idx = b.add(b.add(q_in_lane, b.const_i32(q_subtile*16)), s_lane)
     * Force Python left-to-right SSA. */
    W_lds_idx = rocke_b_add(
        b, rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 16)), ctx->s_lane);

    /* lds_idx = W_lds_idx*BG_cpg + wave_id*cpg + (ch_lane/4)*4
     * Python: b.mul(ch_block_idx, b.const_i32(4)) where ch_block_idx = ch_lane/4. */
    {
        rocke_value_t* mul_wlds = rocke_b_mul(b, W_lds_idx, ctx->c_BG_cpg);
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_cpg);
        rocke_value_t* inner = rocke_b_add(b, mul_wlds, mul_wave);
        rocke_value_t* ch_block_idx = rocke_b_div(b, ctx->ch_lane, rocke_b_const_i32(b, 4));
        rocke_value_t* mul_ch = rocke_b_mul(b, ch_block_idx, rocke_b_const_i32(b, 4));
        lds_idx = rocke_b_add(b, inner, mul_ch);
    }
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    return rocke_b_smem_load_vN_f16(b, lds, indices, 2, 4);
}

/* lds_read_input_s2(q_subtile, lds) -- per-lane <4 x half> read for the s=2 residual.
 * Valid only for c4∈{0,1}; zeroed via select(lane_in_lo_half, vec, fp16x4_zero). */
static rocke_value_t*
    rocke_dconv8c_lds_read_s2(rocke_dconv_8c_ctx_t* ctx, int q_subtile, rocke_value_t* lds)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* W_lds_idx;
    rocke_value_t* lds_idx;
    rocke_value_t* vec;
    rocke_value_t* indices[2];

    /* W_lds_idx = q_in_lane + q_subtile*16 + 2 */
    W_lds_idx = rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 16 + 2));

    {
        rocke_value_t* mul_wlds = rocke_b_mul(b, W_lds_idx, ctx->c_BG_cpg);
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_cpg);
        rocke_value_t* inner = rocke_b_add(b, mul_wlds, mul_wave);
        rocke_value_t* ch_block_idx = rocke_b_div(b, ctx->ch_lane, rocke_b_const_i32(b, 4));
        rocke_value_t* mul_ch = rocke_b_mul(b, ch_block_idx, rocke_b_const_i32(b, 4));
        lds_idx = rocke_b_add(b, inner, mul_ch);
    }
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    vec = rocke_b_smem_load_vN_f16(b, lds, indices, 2, 4);
    return rocke_b_select(b, ctx->lane_in_lo_half, vec, ctx->fp16x4_zero);
}

/* ===================================================================== *
 *  Prologue prefetch  (Python lines 1472-1473)
 * ===================================================================== */
void rocke_dconv8c_prologue_prefetch(rocke_dconv_8c_ctx_t* ctx)
{
    rocke_value_t* vecs[ROCKE_DCONV16C_MAX_PASSES];
    rocke_value_t* lds_idx[ROCKE_DCONV16C_MAX_PASSES];
    int n;

    n = rocke_dconv8c_issue_dram_load(ctx, ctx->c0, vecs, lds_idx, ROCKE_DCONV16C_MAX_PASSES);
    if(n < 0)
    {
        return;
    }
    rocke_dconv8c_store_to_lds(ctx, vecs, lds_idx, n, ctx->A_smem);
    rocke_b_sync(ctx->b);
}

/* ===================================================================== *
 *  H-row streaming loop  (Python lines 1483-1537)
 * ===================================================================== */
rocke_kernel_def_t* rocke_dconv8c_stream_h_loop(rocke_dconv_8c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int KH = p->KH;
    int y;
    int n_iters = ctx->n_iters;
    int q_subtiles = ctx->q_subtiles;

    /* acc_tiles seeded to zero_acc (q_subtiles x KH). */
    {
        int qt, slot;
        for(qt = 0; qt < q_subtiles; ++qt)
        {
            for(slot = 0; slot < KH; ++slot)
            {
                ctx->acc_tiles[qt][slot] = ctx->zero_acc;
            }
        }
    }

    for(y = 0; y < n_iters; ++y)
    {
        rocke_value_t* cur;
        rocke_value_t* nxt;
        /* Inputs: per-q (main folded + s2 residual). */
        rocke_value_t* in_main[ROCKE_DCONV_MAX_QTILES];
        rocke_value_t* in_s2[ROCKE_DCONV_MAX_QTILES];
        rocke_value_t* loads_next_vecs[ROCKE_DCONV16C_MAX_PASSES];
        rocke_value_t* loads_next_lds[ROCKE_DCONV16C_MAX_PASSES];
        int n_loads_next = 0;
        bool has_loads_next = false;
        int qt;
        int p_flush_val;
        int P_FLUSH;

        if((y % 2 == 0) || !ctx->spec->double_buffer)
        {
            cur = ctx->A_smem;
            nxt = ctx->B_smem;
        }
        else
        {
            cur = ctx->B_smem;
            nxt = ctx->A_smem;
        }

        /* Read inputs from cur (Python lines 1486-1487). */
        for(qt = 0; qt < q_subtiles; ++qt)
        {
            in_main[qt] = rocke_dconv8c_lds_read_main(ctx, qt, cur);
            in_s2[qt] = rocke_dconv8c_lds_read_s2(ctx, qt, cur);
        }

        /* Issue next-row DRAM loads (Python lines 1488-1490). */
        if(y + 1 < n_iters)
        {
            n_loads_next = rocke_dconv8c_issue_dram_load(ctx,
                                                         rocke_b_const_i32(b, y + 1),
                                                         loads_next_vecs,
                                                         loads_next_lds,
                                                         ROCKE_DCONV16C_MAX_PASSES);
            if(n_loads_next < 0)
            {
                return NULL;
            }
            has_loads_next = true;
        }

        /* MFMA chain (Python lines 1491-1503):
         * for qt: for r: main atom + residual atom into circular slot. */
        for(qt = 0; qt < q_subtiles; ++qt)
        {
            int r_const;
            for(r_const = 0; r_const < KH; ++r_const)
            {
                int p_idx = (((y - r_const) % KH) + KH) % KH;
                rocke_value_t* acc_in = ctx->acc_tiles[qt][p_idx];

                /* acc_in = mfma_16x16x16(weights_main[r], in_main, acc_in) */
                acc_in = rocke_b_mfma_f32_16x16x16_f16(
                    b, ctx->weights_main[r_const], in_main[qt], acc_in);
                /* acc_in = mfma_16x16x16(weights_s2[r], in_s2, acc_in) */
                acc_in
                    = rocke_b_mfma_f32_16x16x16_f16(b, ctx->weights_s2[r_const], in_s2[qt], acc_in);
                ctx->acc_tiles[qt][p_idx] = acc_in;
            }
        }

        /* Store next-row loads (Python lines 1505-1508). */
        if(has_loads_next)
        {
            if(!ctx->spec->double_buffer)
            {
                rocke_b_sync(b);
            }
            rocke_dconv8c_store_to_lds(ctx, loads_next_vecs, loads_next_lds, n_loads_next, nxt);
        }
        rocke_b_sync(b);

        p_flush_val = y - (KH - 1);
        P_FLUSH = ((p_flush_val % KH) + KH) % KH;

        /* Flush slot to D (Python lines 1512-1535). */
        if(0 <= p_flush_val && p_flush_val < p->H)
        {
            for(qt = 0; qt < q_subtiles; ++qt)
            {
                rocke_value_t* acc_to_flush = ctx->acc_tiles[qt][P_FLUSH];
                rocke_value_t* out_q;
                rocke_value_t* out_q_valid;
                rocke_value_t* k_val;
                rocke_value_t* c4_valid;
                rocke_value_t* d_base;
                rocke_value_t* d_valid;
                rocke_value_t* d_base_bytes;
                rocke_value_t* store_valid;
                rocke_value_t* safe_d_off;
                rocke_value_t* acc_h;
                const char* off_names[4];
                rocke_value_t* off_vals[4];

                /* out_q = q_tile_start + qt*16 + q_in_lane */
                out_q
                    = rocke_b_add(b,
                                  rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, qt * 16)),
                                  ctx->q_in_lane);
                out_q_valid = rocke_b_cmp_lt(b, out_q, ctx->c_W);

                /* k_val = g*kpg + c4*4 (force Python left-to-right SSA). */
                {
                    rocke_value_t* mul_g = rocke_b_mul(b, ctx->g, ctx->c_kpg);
                    rocke_value_t* mul_c4 = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, 4));
                    k_val = rocke_b_add(b, mul_g, mul_c4);
                }

                /* Gate: only store for c4 < kpg/4 = 2 (Python: b.cmp_lt(c4, const(kpg//4))). */
                c4_valid = rocke_b_cmp_lt(b, ctx->c4, rocke_b_const_i32(b, ctx->p.kpg / 4));

                off_names[0] = "n";
                off_vals[0] = ctx->n;
                off_names[1] = "h";
                off_vals[1] = rocke_b_const_i32(b, p_flush_val);
                off_names[2] = "w";
                off_vals[2] = out_q;
                off_names[3] = "k";
                off_vals[3] = k_val;
                if(!rocke_transforms_descriptor_offset(
                       b, ctx->d_desc, off_names, off_vals, 4, &d_base, &d_valid))
                {
                    return NULL;
                }

                d_base_bytes = rocke_b_mul(b, d_base, ctx->c_half_bytes);
                /* store_valid = b.land(out_q_valid, c4_valid) */
                store_valid = rocke_b_land(b, out_q_valid, c4_valid);
                safe_d_off = rocke_b_select(b, store_valid, d_base_bytes, ctx->oob_sentinel);
                acc_h = rocke_b_vec_trunc_f32_to_f16(b, acc_to_flush);
                rocke_b_buffer_store_vN_f16(b, ctx->d_rsrc, safe_d_off, ctx->c0, acc_h, 2);
            }
        }

        /* Unconditional slot reset. */
        for(qt = 0; qt < q_subtiles; ++qt)
        {
            ctx->acc_tiles[qt][P_FLUSH] = ctx->zero_acc;
        }
    }

    return rocke_ir_builder_kernel(b);
}
