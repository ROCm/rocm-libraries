// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_build_32c.c -- C99 port of build_direct_conv_32c
 * (rocke/instances/common/conv_direct_grouped.py, lines 1657-1975).
 *
 * The 32c kernel uses mfma_f32_32x32x8_f16 (M=32=kpg, K=8).
 * Per (r, s): 4 consecutive MFMA calls (atom_idx=0..3, ch_start=0,8,16,24).
 * Accumulator: <16 x float> (32*32/64 = 16 slots per lane).
 * LDS double-buffered; same ping-pong scheme as 16c/8c.
 * Lane decomposition:
 *   q_in_lane = lane % 32 → M row (k_out within group) + N column (output W).
 *   k_blk     = lane / 32 → K-block (0→ch=0..3, 1→ch=4..7 within each atom).
 *
 * Phase functions: prologue, load_weights, build_chunk_meta, build_descriptors,
 * prologue_prefetch, stream_h_loop.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  Prologue  (Python lines 1676-1733)
 * ===================================================================== */
bool rocke_dconv32c_prologue(rocke_dconv_32c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_32c_spec_t* spec = ctx->spec;
    char reason[ROCKE_ERR_MSG_CAP];

    if(rocke_direct_conv_32c_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    if(!rocke_direct_conv_32c_is_valid_spec(spec, ctx->arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->p = spec->problem;

    ctx->BLOCK_Q = spec->block_q;
    ctx->BLOCK_GROUPS = spec->block_groups;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_conv_32c_threads_per_block(spec);
    ctx->LDS_W = ctx->BLOCK_Q + ctx->p.KW - 1;
    ctx->LDS_ROW_FP16 = ctx->LDS_W * ctx->BLOCK_GROUPS * ctx->p.cpg;
    ctx->LOAD_VEC = 4;
    ctx->NUM_VEC4 = ctx->LDS_ROW_FP16 / ctx->LOAD_VEC;
    ctx->N_CH_BLOCKS = ctx->p.cpg / ctx->LOAD_VEC; /* = 8 for cpg=32 */
    if(ctx->NUM_VEC4 == 0)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    ctx->PASSES = (ctx->NUM_VEC4 + ctx->THREADS - 1) / ctx->THREADS;
    ctx->q_subtiles = ctx->BLOCK_Q / 32;
    ctx->n_iters = ctx->p.H + ctx->p.KH - 1;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

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

    /* Lane decomposition for mfma_f32_32x32x8_f16 (Python lines 1715-1716). */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);
    ctx->q_in_lane = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, 32));
    ctx->k_blk = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, 32));

    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->n = rocke_b_block_id_z(b);
    ctx->g_tile = ctx->by;
    ctx->g = rocke_b_add(b, rocke_b_mul(b, ctx->g_tile, ctx->c_BG), ctx->wave_id);
    ctx->q_tile_start = rocke_b_mul(b, ctx->bx, ctx->c_BQ);

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
    /* zero_acc = b.zero_vec_f32(16) — 16 f32 elements (32*32/64) */
    ctx->zero_acc = rocke_b_zero_vec_f32(b, 16);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Weight-load phase  (Python lines 1738-1763)
 *
 *  Per (r, s, atom_idx): each lane loads 4 f16 at
 *    B[k_out_val, r, s, ch_start + k_blk*4 .. ch_start + k_blk*4 + 3]
 *  where ch_start = atom_idx * 8.
 * ===================================================================== */
void rocke_dconv32c_load_weights(rocke_dconv_32c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);
    int r_const, s_const, atom_idx;

    {
        int lengths[4];
        static const char* const coord_names[4] = {"k_out", "r", "s", "c"};
        lengths[0] = total_k;
        lengths[1] = ctx->p.KH;
        lengths[2] = ctx->p.KW;
        lengths[3] = ctx->p.cpg;
        ctx->b_desc = rocke_tensor_descriptor_naive(b, "B", lengths, 4, NULL, coord_names, 4);
    }

    ctx->k_out_val = rocke_b_add(b, rocke_b_mul(b, ctx->g, ctx->c_kpg), ctx->q_in_lane);
    /* ch_in_atom = k_blk * 4: emitted here (after k_out_val) to match Python op order. */
    ctx->ch_in_atom = rocke_b_mul(b, ctx->k_blk, rocke_b_const_i32(b, 4));

    ctx->n_weight_r = ctx->p.KH;
    ctx->n_weight_s = ctx->p.KW;

    for(r_const = 0; r_const < ctx->p.KH; ++r_const)
    {
        for(s_const = 0; s_const < ctx->p.KW; ++s_const)
        {
            for(atom_idx = 0; atom_idx < 4; ++atom_idx)
            {
                int ch_start = atom_idx * 8;
                rocke_value_t* ch_off;
                rocke_value_t* w_off = NULL;
                rocke_value_t* valid = NULL;
                const char* in_names[4] = {"k_out", "r", "s", "c"};
                rocke_value_t* in_values[4];

                /* ch_off = ch_start + k_blk*4 (= ch_start + ch_in_atom)
                 * Python: b.add(b.const_i32(ch_start), ch_in_atom)
                 * Force Python left-to-right: const first, then add. */
                ch_off = rocke_b_add(b, rocke_b_const_i32(b, ch_start), ctx->ch_in_atom);

                in_values[0] = ctx->k_out_val;
                in_values[1] = rocke_b_const_i32(b, r_const);
                in_values[2] = rocke_b_const_i32(b, s_const);
                in_values[3] = ch_off;
                rocke_transforms_descriptor_offset(
                    b, ctx->b_desc, in_names, in_values, 4, &w_off, &valid);
                ctx->weights[r_const][s_const][atom_idx] = rocke_b_buffer_load_vN_f16(
                    b, ctx->b_rsrc, rocke_b_mul(b, w_off, ctx->c_half_bytes), ctx->c0, 2);
            }
        }
    }
}

/* ===================================================================== *
 *  Chunk-decode phase  (Python lines 1765-1795)
 *
 *  chunk_desc with N_CH_BLOCKS = cpg/4 = 8 (for cpg=32).
 * ===================================================================== */
void rocke_dconv32c_build_chunk_meta(rocke_dconv_32c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    static const char* const coord_names[3] = {"W_lds", "group_in_wg", "ch_block"};
    int lengths[3];
    int pass_idx;
    rocke_tensor_descriptor_t* naive;

    lengths[0] = ctx->LDS_W;
    lengths[1] = ctx->BLOCK_GROUPS;
    lengths[2] = ctx->N_CH_BLOCKS; /* = 8 for cpg=32 */

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
 *  Descriptor phase  (Python lines 1796-1870)
 * ===================================================================== */
void rocke_dconv32c_build_descriptors(rocke_dconv_32c_ctx_t* ctx)
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
 *  DRAM load / LDS store helpers
 * ===================================================================== */
static int rocke_dconv32c_issue_dram_load(rocke_dconv_32c_ctx_t* ctx,
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

static void rocke_dconv32c_store_to_lds(rocke_dconv_32c_ctx_t* ctx,
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

/* lds_read_input(q_subtile, s_const, atom_idx, lds):
 * ch_start = atom_idx*8; ch_off = ch_start + k_blk*4;
 * W_lds_idx = q_in_lane + q_subtile*32 + s_const;
 * lds_idx = W_lds_idx*BG_cpg + wave_id*cpg + ch_off. */
static rocke_value_t* rocke_dconv32c_lds_read_input(
    rocke_dconv_32c_ctx_t* ctx, int q_subtile, int s_const, int atom_idx, rocke_value_t* lds)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* W_lds_idx;
    rocke_value_t* lds_idx;
    rocke_value_t* indices[2];
    int ch_start = atom_idx * 8;
    rocke_value_t* ch_off;

    /* W_lds_idx = q_in_lane + (q_subtile*32 + s_const) */
    W_lds_idx = rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 32 + s_const));

    /* ch_off = ch_start + k_blk*4
     * Python: b.add(b.const_i32(ch_start), b.mul(k_blk, b.const_i32(4)))
     * Emit const(ch_start) first, then const(4)/mul, to match Python left-to-right order. */
    {
        rocke_value_t* c_ch_start = rocke_b_const_i32(b, ch_start);
        rocke_value_t* mul_k = rocke_b_mul(b, ctx->k_blk, rocke_b_const_i32(b, 4));
        ch_off = rocke_b_add(b, c_ch_start, mul_k);
    }

    /* lds_idx = W_lds_idx*BG_cpg + wave_id*cpg + ch_off */
    {
        rocke_value_t* mul_wlds = rocke_b_mul(b, W_lds_idx, ctx->c_BG_cpg);
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_cpg);
        rocke_value_t* inner = rocke_b_add(b, mul_wlds, mul_wave);
        lds_idx = rocke_b_add(b, inner, ch_off);
    }
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    return rocke_b_smem_load_vN_f16(b, lds, indices, 2, 4);
}

/* ===================================================================== *
 *  Prologue prefetch  (Python lines 1861-1862)
 * ===================================================================== */
void rocke_dconv32c_prologue_prefetch(rocke_dconv_32c_ctx_t* ctx)
{
    rocke_value_t* vecs[ROCKE_DCONV16C_MAX_PASSES];
    rocke_value_t* lds_idx[ROCKE_DCONV16C_MAX_PASSES];
    int n;

    n = rocke_dconv32c_issue_dram_load(ctx, ctx->c0, vecs, lds_idx, ROCKE_DCONV16C_MAX_PASSES);
    if(n < 0)
    {
        return;
    }
    rocke_dconv32c_store_to_lds(ctx, vecs, lds_idx, n, ctx->A_smem);
    rocke_b_sync(ctx->b);
}

/* ===================================================================== *
 *  H-row streaming loop  (Python lines 1872-1944)
 * ===================================================================== */
rocke_kernel_def_t* rocke_dconv32c_stream_h_loop(rocke_dconv_32c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int KH = p->KH;
    int KW = p->KW;
    int y;
    int n_iters = ctx->n_iters;
    int q_subtiles = ctx->q_subtiles;

    /* acc_tiles seeded to zero_acc (q_subtiles x KH; each = <16 x float>). */
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
        /* inputs_by_q[qt][s][atom]: per-lane <4 x half> inputs. */
        rocke_value_t* inputs_by_q[ROCKE_DCONV_MAX_QTILES][ROCKE_DCONV32C_MAX_KW]
                                  [ROCKE_DCONV32C_MAX_ATOMS];
        rocke_value_t* loads_next_vecs[ROCKE_DCONV16C_MAX_PASSES];
        rocke_value_t* loads_next_lds[ROCKE_DCONV16C_MAX_PASSES];
        int n_loads_next = 0;
        bool has_loads_next = false;
        int qt, r_const, s_const, atom_idx;
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

        /* Read all inputs from cur (Python: inputs_by_q comprehension). */
        for(qt = 0; qt < q_subtiles; ++qt)
        {
            for(s_const = 0; s_const < KW; ++s_const)
            {
                for(atom_idx = 0; atom_idx < 4; ++atom_idx)
                {
                    inputs_by_q[qt][s_const][atom_idx]
                        = rocke_dconv32c_lds_read_input(ctx, qt, s_const, atom_idx, cur);
                }
            }
        }

        if(y + 1 < n_iters)
        {
            n_loads_next = rocke_dconv32c_issue_dram_load(ctx,
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

        /* MFMA chain (Python lines 1885-1896):
         * for qt: for r: for s: for atom: 4 mfma_f32_32x32x8_f16 into slot. */
        for(qt = 0; qt < q_subtiles; ++qt)
        {
            for(r_const = 0; r_const < KH; ++r_const)
            {
                int p_idx = (((y - r_const) % KH) + KH) % KH;
                rocke_value_t* acc_in = ctx->acc_tiles[qt][p_idx];

                for(s_const = 0; s_const < KW; ++s_const)
                {
                    for(atom_idx = 0; atom_idx < 4; ++atom_idx)
                    {
                        acc_in
                            = rocke_b_mfma_f32_32x32x8_f16(b,
                                                           ctx->weights[r_const][s_const][atom_idx],
                                                           inputs_by_q[qt][s_const][atom_idx],
                                                           acc_in);
                    }
                }
                ctx->acc_tiles[qt][p_idx] = acc_in;
            }
        }

        if(has_loads_next)
        {
            if(!ctx->spec->double_buffer)
            {
                rocke_b_sync(b);
            }
            rocke_dconv32c_store_to_lds(ctx, loads_next_vecs, loads_next_lds, n_loads_next, nxt);
        }
        rocke_b_sync(b);

        p_flush_val = y - (KH - 1);
        P_FLUSH = ((p_flush_val % KH) + KH) % KH;

        /* Flush (Python lines 1905-1943):
         * 4 octants × 2 halves → 8 vec2 stores per qt.
         * slot i: row = (i//4)*8 + k_blk*4 + (i%4), col = q_in_lane.
         * k_base = g*kpg + k_blk*4.
         * For each octant (o) and half (h):
         *   slot pair: o*4+h*2, o*4+h*2+1
         *   row_off   = o*8 + h*2
         *   k_val     = k_base + row_off
         *   store as vec2 (b.buffer_store_vN_f16(..., 1)). */
        if(0 <= p_flush_val && p_flush_val < p->H)
        {
            for(qt = 0; qt < q_subtiles; ++qt)
            {
                rocke_value_t* acc_to_flush = ctx->acc_tiles[qt][P_FLUSH];
                rocke_value_t* out_q;
                rocke_value_t* out_q_valid;
                rocke_value_t* k_base;
                int octant, half;

                /* out_q = q_tile_start + qt*32 + q_in_lane */
                out_q
                    = rocke_b_add(b,
                                  rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, qt * 32)),
                                  ctx->q_in_lane);
                out_q_valid = rocke_b_cmp_lt(b, out_q, ctx->c_W);

                /* k_base = g*kpg + k_blk*4 (Python: b.add(b.mul(g,c_kpg), b.mul(k_blk,4)))
                 * Force Python left-to-right SSA. */
                {
                    rocke_value_t* mul_g = rocke_b_mul(b, ctx->g, ctx->c_kpg);
                    rocke_value_t* mul_kb = rocke_b_mul(b, ctx->k_blk, rocke_b_const_i32(b, 4));
                    k_base = rocke_b_add(b, mul_g, mul_kb);
                }

                for(octant = 0; octant < 4; ++octant)
                {
                    for(half = 0; half < 2; ++half)
                    {
                        int slot0 = octant * 4 + half * 2;
                        int slot1 = slot0 + 1;
                        int row_off = octant * 8 + half * 2;
                        rocke_value_t* k_val;
                        rocke_value_t* d_base;
                        rocke_value_t* d_valid;
                        rocke_value_t* d_base_bytes;
                        rocke_value_t* safe_d_off;
                        rocke_value_t* e0;
                        rocke_value_t* e1;
                        rocke_value_t* pair_elems[2];
                        rocke_value_t* pair_f32;
                        rocke_value_t* pair_h;
                        const char* off_names[4];
                        rocke_value_t* off_vals[4];

                        k_val = rocke_b_add(b, k_base, rocke_b_const_i32(b, row_off));

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
                        safe_d_off
                            = rocke_b_select(b, out_q_valid, d_base_bytes, ctx->oob_sentinel);

                        /* e0, e1 = vec_extract(acc, slot0/slot1)
                         * pair_f32 = vec_pack([e0, e1], F32)
                         * pair_h = vec_trunc_f32_to_f16(pair_f32)
                         * buffer_store_vN_f16(..., 1)  -- 1 dword = 2 f16 */
                        e0 = rocke_b_vec_extract(b, acc_to_flush, slot0);
                        e1 = rocke_b_vec_extract(b, acc_to_flush, slot1);
                        pair_elems[0] = e0;
                        pair_elems[1] = e1;
                        pair_f32 = rocke_b_vec_pack(b, pair_elems, 2, rocke_f32());
                        pair_h = rocke_b_vec_trunc_f32_to_f16(b, pair_f32);
                        rocke_b_buffer_store_vN_f16(b, ctx->d_rsrc, safe_d_off, ctx->c0, pair_h, 1);
                    }
                }
            }
        }

        for(qt = 0; qt < q_subtiles; ++qt)
        {
            ctx->acc_tiles[qt][P_FLUSH] = ctx->zero_acc;
        }
    }

    return rocke_ir_builder_kernel(b);
}
