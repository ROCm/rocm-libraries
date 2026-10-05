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
#include <string.h>

#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include <stdio.h>

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

    /* io_type = _io_type(p.dtype): f16 or bf16 IR type. */
    ctx->io_type = rocke_b_io_ir_type(b, ctx->p.dtype ? ctx->p.dtype : "fp16");
    if(ctx->io_type == NULL)
    {
        return false; /* builder sticky error already set by rocke_b_io_ir_type */
    }

    ctx->BLOCK_Q = spec->block_q;
    ctx->BLOCK_GROUPS = spec->block_groups;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_conv_32c_threads_per_block(spec);
    ctx->LDS_W = (ctx->BLOCK_Q - 1) * ctx->p.stride + ctx->p.KW;
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

    /* params: the AOT kernarg block, in conv_abi order. */
    rocke_dconv_emit_params(b, &ctx->params, "fwd", ctx->io_type);
    ctx->A = ctx->params.A;
    ctx->Bp = ctx->params.Bp;
    ctx->D = ctx->params.D;
    ctx->A_bytes = ctx->params.A_bytes;
    ctx->B_bytes = ctx->params.B_bytes;
    ctx->D_bytes = ctx->params.D_bytes;

    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    ctx->c_BG = rocke_b_const_i32(b, ctx->BLOCK_GROUPS);
    ctx->c_BQ = rocke_b_const_i32(b, ctx->BLOCK_Q);
    ctx->c_cpg = rocke_b_const_i32(b, ctx->p.cpg);
    ctx->c_kpg = rocke_b_const_i32(b, ctx->p.kpg);
    /* AOT: the store guard bounds against the runtime output width. */
    ctx->c_W = ctx->params.p_Wo;
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

    ctx->lds_total_elems = ctx->PASSES * ctx->THREADS * ctx->LOAD_VEC;
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = ctx->lds_total_elems;
        ctx->A_smem = rocke_b_smem_alloc(b, ctx->io_type, shape, 2, "lds_a");
        if(spec->double_buffer)
        {
            ctx->B_smem = rocke_b_smem_alloc(b, ctx->io_type, shape, 2, "lds_b");
        }
        else
        {
            ctx->B_smem = ctx->A_smem;
        }
    }

    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);
    ctx->d_rsrc = rocke_b_buffer_rsrc(b, ctx->D, ctx->D_bytes);

    ctx->io_vec4_zero = rocke_b_zero_vec(b, ctx->io_type, 4);
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
    const int is_bf16 = ctx->p.dtype && strcmp(ctx->p.dtype, "bf16") == 0;
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
                if(is_bf16)
                    ctx->weights[r_const][s_const][atom_idx] = rocke_b_buffer_load_vN_bf16(
                        b, ctx->b_rsrc, rocke_b_mul(b, w_off, ctx->c_half_bytes), ctx->c0, 2);
                else
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
    rocke_dynamic_tensor_descriptor_t* a_dyn;

    /* AOT: every extent and base stride comes from the kernarg block; PAD and
     * stride stay build-time. The output descriptor is built inside the
     * streaming-loop phase, where Python builds it. */
    a_dyn = rocke_dconv_a_descriptor_dynamic(
        b, &ctx->params, ctx->p.PAD, ctx->p.stride, "q_pos", "W_lds_pos");
    if(!a_dyn)
        return;
    ctx->a_desc = &a_dyn->base;
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
    const int is_bf16 = ctx->p.dtype && strcmp(ctx->p.dtype, "bf16") == 0;
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
        if(is_bf16)
            a_vec = rocke_b_buffer_load_vN_bf16(b, ctx->a_rsrc, safe_off, ctx->c0, 2);
        else
            a_vec = rocke_b_buffer_load_vN_f16(b, ctx->a_rsrc, safe_off, ctx->c0, 2);
        a_vec = rocke_b_select(b, valid, a_vec, ctx->io_vec4_zero);
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
        rocke_b_smem_store_vN(b, lds, indices, 2, vecs[i], 4);
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

    /* W_lds_idx = b.add(
     *     b.mul(b.add(q_in_lane, b.const_i32(q_subtile*32)), b.const_i32(c_stride_32c)),
     *     b.const_i32(s_const))
     * Force Python left-to-right SSA order: inner add first, then mul, then outer add. */
    {
        rocke_value_t* base = rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 32));
        rocke_value_t* c_s = rocke_b_const_i32(b, ctx->p.stride);
        rocke_value_t* mul_v = rocke_b_mul(b, base, c_s);
        rocke_value_t* c_sc = rocke_b_const_i32(b, s_const);
        W_lds_idx = rocke_b_add(b, mul_v, c_sc);
    }

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
    return rocke_b_smem_load_vN(b, lds, indices, 2, ctx->io_type, 4);
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
    const int is_bf16 = ctx->p.dtype && strcmp(ctx->p.dtype, "bf16") == 0;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int KH = p->KH;
    int KW = p->KW;
    int j, qt;
    int q_subtiles = ctx->q_subtiles;
    int num_accs = q_subtiles * KH;
    int UNROLL;
    rocke_value_t* c_unroll;
    rocke_value_t* n_iters_v;
    rocke_iter_arg_t iter_args[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS];
    char acc_names[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS][32];
    rocke_value_t* accs_flat[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS];
    rocke_for_t for_op;
    rocke_value_t* y_base;

    /* AOT: Hi is a kernarg. The body is unrolled lcm(2, KH) times so the LDS
     * ping-pong (period 2) and the accumulator slot rotation (period KH) both
     * line up with the body boundary. */
    UNROLL = rocke_dconv_row_loop_unroll(KH, /*lds_ping_pong=*/true);
    c_unroll = rocke_b_const_i32(b, UNROLL);
    n_iters_v = rocke_b_add(b, ctx->params.p_Hi, rocke_b_const_i32(b, KH - 1));

    {
        rocke_dynamic_tensor_descriptor_t* d_dyn
            = rocke_dconv_d_descriptor_dynamic(b, &ctx->params);
        if(!d_dyn)
            return NULL;
        ctx->d_desc = &d_dyn->base;
    }

    if(num_accs > (int)(sizeof(iter_args) / sizeof(iter_args[0])))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE; /* too many accumulator slots */
        return NULL;
    }
    for(qt = 0; qt < q_subtiles; ++qt)
    {
        int pi;
        for(pi = 0; pi < KH; ++pi)
        {
            int idx = qt * KH + pi;
            snprintf(acc_names[idx], sizeof(acc_names[0]), "acc_q%d_p%d", qt, pi);
            iter_args[idx].name = acc_names[idx];
            iter_args[idx].init = ctx->zero_acc;
        }
    }

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->c0,
                                  n_iters_v,
                                  c_unroll,
                                  iter_args,
                                  num_accs,
                                  "y_row",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/false);
    y_base = for_op.iv;
    {
        int i;
        for(i = 0; i < for_op.num_iter_vars; ++i)
            accs_flat[i] = for_op.iter_vars[i];
    }

    rocke_b_region_enter(b, for_op.body);
    for(j = 0; j < UNROLL; ++j)
    {
        rocke_value_t* cur;
        rocke_value_t* nxt;
        rocke_value_t* inputs_by_q[ROCKE_DCONV_MAX_QTILES][ROCKE_DCONV32C_MAX_KW]
                                  [ROCKE_DCONV32C_MAX_ATOMS];
        rocke_value_t* loads_next_vecs[ROCKE_DCONV16C_MAX_PASSES];
        rocke_value_t* loads_next_lds[ROCKE_DCONV16C_MAX_PASSES];
        int n_loads_next;
        int r_const, s_const, atom_idx;
        int P_FLUSH;
        rocke_value_t* p_flush_v;
        rocke_value_t* row_ok;
        rocke_value_t* ho_row_v;

        if((j % 2 == 0) || !ctx->spec->double_buffer)
        {
            cur = ctx->A_smem;
            nxt = ctx->B_smem;
        }
        else
        {
            cur = ctx->B_smem;
            nxt = ctx->A_smem;
        }

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

        /* Rows past the image zero-fill through the descriptor bound. */
        n_loads_next
            = rocke_dconv32c_issue_dram_load(ctx,
                                             rocke_b_add(b, y_base, rocke_b_const_i32(b, j + 1)),
                                             loads_next_vecs,
                                             loads_next_lds,
                                             ROCKE_DCONV16C_MAX_PASSES);
        if(n_loads_next < 0)
            return NULL;

        for(qt = 0; qt < q_subtiles; ++qt)
        {
            for(r_const = 0; r_const < KH; ++r_const)
            {
                /* y_base is 0 mod KH, so the rotating slot collapses to j. */
                int p_idx = (((j - r_const) % KH) + KH) % KH;
                int flat = qt * KH + p_idx;
                rocke_value_t* acc_in = accs_flat[flat];

                for(s_const = 0; s_const < KW; ++s_const)
                {
                    for(atom_idx = 0; atom_idx < 4; ++atom_idx)
                    {
                        if(is_bf16)
                            acc_in = rocke_b_mfma_f32_32x32x8_bf16(
                                b,
                                ctx->weights[r_const][s_const][atom_idx],
                                inputs_by_q[qt][s_const][atom_idx],
                                acc_in);
                        else
                            acc_in = rocke_b_mfma_f32_32x32x8_f16(
                                b,
                                ctx->weights[r_const][s_const][atom_idx],
                                inputs_by_q[qt][s_const][atom_idx],
                                acc_in);
                    }
                }
                accs_flat[flat] = acc_in;
            }
        }

        /* Single-buffer read-after-write guard; see the 16c builder. */
        if(!ctx->spec->double_buffer)
            rocke_b_sync(b);
        rocke_dconv32c_store_to_lds(ctx, loads_next_vecs, loads_next_lds, n_loads_next, nxt);
        rocke_b_sync(b);

        P_FLUSH = (((j - (KH - 1)) % KH) + KH) % KH;
        p_flush_v = rocke_b_add(b, y_base, rocke_b_const_i32(b, j - (KH - 1)));
        {
            rocke_value_t* ge = rocke_b_cmp_ge(b, p_flush_v, ctx->c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, p_flush_v, ctx->params.p_Hi);
            row_ok = rocke_b_land(b, ge, lt);
        }
        if(p->stride > 1)
        {
            rocke_value_t* c_stride_v = rocke_b_const_i32(b, p->stride);
            rocke_value_t* md = rocke_b_mod(b, p_flush_v, c_stride_v);
            rocke_value_t* eq = rocke_b_cmp_eq(b, md, ctx->c0);
            row_ok = rocke_b_land(b, row_ok, eq);
            ho_row_v = rocke_b_div(b, p_flush_v, c_stride_v);
        }
        else
        {
            ho_row_v = p_flush_v;
        }
        ho_row_v = rocke_b_select(b, row_ok, ho_row_v, ctx->c0);

        for(qt = 0; qt < q_subtiles; ++qt)
        {
            int flat = qt * KH + P_FLUSH;
            rocke_value_t* acc_to_flush = accs_flat[flat];
            rocke_value_t* out_q;
            rocke_value_t* store_ok;
            rocke_value_t* k_base;
            int octant, half;

            {
                rocke_value_t* inner
                    = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, qt * 32));
                out_q = rocke_b_add(b, inner, ctx->q_in_lane);
            }
            store_ok = rocke_b_land(b, row_ok, rocke_b_cmp_lt(b, out_q, ctx->c_W));

            /* C output layout for mfma_f32_32x32x8_f16 (wave64), 16 slots:
             *   slot i: row = (i//4)*8 + k_blk*4 + (i%4), col = q_in_lane */
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
                    rocke_value_t* pair_elems[2];
                    rocke_value_t* pair_f32;
                    rocke_value_t* pair_h;
                    const char* off_names[4];
                    rocke_value_t* off_vals[4];

                    k_val = rocke_b_add(b, k_base, rocke_b_const_i32(b, row_off));

                    off_names[0] = "n";
                    off_vals[0] = ctx->n;
                    off_names[1] = "h";
                    off_vals[1] = ho_row_v;
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
                    safe_d_off = rocke_b_select(b, store_ok, d_base_bytes, ctx->oob_sentinel);

                    pair_elems[0] = rocke_b_vec_extract(b, acc_to_flush, slot0);
                    pair_elems[1] = rocke_b_vec_extract(b, acc_to_flush, slot1);
                    pair_f32 = rocke_b_vec_pack(b, pair_elems, 2, rocke_f32());
                    if(is_bf16)
                    {
                        pair_h = rocke_b_vec_trunc_f32_to_bf16(b, pair_f32);
                        rocke_b_buffer_store_vN_bf16(
                            b, ctx->d_rsrc, safe_d_off, ctx->c0, pair_h, 1);
                    }
                    else
                    {
                        pair_h = rocke_b_vec_trunc_f32_to_f16(b, pair_f32);
                        rocke_b_buffer_store_vN_f16(b, ctx->d_rsrc, safe_d_off, ctx->c0, pair_h, 1);
                    }
                }
            }
            accs_flat[flat] = ctx->zero_acc;
        }
    }
    rocke_b_scf_yield(b, accs_flat, num_accs);
    rocke_b_region_leave(b);

    return rocke_ir_builder_kernel(b);
}
