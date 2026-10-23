// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_build_16c_loop_closures.c
 *
 * Chunked port of rocke/instances/common/conv_direct_grouped.py, covering the
 * 16c closures + the unrolled H-row streaming loop:
 *
 *   Python                                C99 (this TU)
 *   -----------------------------------   ------------------------------------
 *   def issue_dram_load(...)  (521-562)    rocke_dconv16c_issue_dram_load
 *   def store_to_lds(...)     (564-566)    rocke_dconv16c_store_to_lds
 *   def lds_read_input(...)   (570-590)    rocke_dconv16c_lds_read_input
 *   def lds_read_input_k32(.) (592-607)    rocke_dconv16c_lds_read_input_k32
 *   prologue prefetch         (609-616)    rocke_dconv16c_prologue_prefetch
 *   the H-row streaming loop  (618-740)    rocke_dconv16c_stream_h_loop
 *
 * The closures captured the enclosing-function locals; here they read/write
 * exactly the ctx fields the internal header carries, and emit IR in
 * byte-identical Python builder-call order. Peer phases (prologue, weights,
 * chunk_meta, descriptors) are declared in the internal header and resolved at
 * link time.
 */
#include <stdio.h>

#include "rocke/instance_conv_direct_grouped_internal.h"

#include <string.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  Closure: issue_dram_load(y_iter_val)            (Python lines 521-562)
 *
 *  Per-thread DRAM read of one vec4 of A. Returns `(vec4, lds_idx)` pairs (one
 *  per chunk_meta entry). The caller decides when to store them to LDS so the v6
 *  pipeline can issue the next-row reads before the current-row MFMAs.
 * ===================================================================== */
int rocke_dconv16c_issue_dram_load(rocke_dconv_16c_ctx_t* ctx,
                                   rocke_value_t* y_iter_val,
                                   rocke_value_t** out_vecs,
                                   rocke_value_t** out_lds_idx,
                                   int out_cap)
{
    rocke_ir_builder_t* b = ctx->b;
    const int is_bf16 = ctx->p.dtype && strcmp(ctx->p.dtype, "bf16") == 0;
    int count = 0;
    int i;

    /* out = [] ; for cm in chunk_meta: */
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

        /* c_val = b.add(b.mul(cm["abs_group"], c_cpg),
         *               b.mul(cm["ch_block"], b.const_i32(4)))
         * Python evaluates b.add's first arg (the abs_group*cpg mul) before its
         * second arg (the ch_block*4 mul). C arg eval order is unspecified, so
         * hoist each mul into a temp in left-to-right order. */
        {
            rocke_value_t* mul_ag = rocke_b_mul(b, ctx->chunk_meta[i].abs_group, ctx->c_cpg);
            rocke_value_t* mul_cb
                = rocke_b_mul(b, ctx->chunk_meta[i].ch_block, rocke_b_const_i32(b, 4));
            c_val = rocke_b_add(b, mul_ag, mul_cb);
        }

        /* a_off_elems, addr_valid = a_desc.offset(b, n=n, y_iter=y_iter_val,
         *      q_pos=q_tile_start, W_lds_pos=cm["W_lds"], c=c_val) */
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

        /* valid = b.land(addr_valid, cm["in_bounds"]) */
        valid = rocke_b_land(b, addr_valid, ctx->chunk_meta[i].in_bounds);
        /* a_off_bytes = b.mul(a_off_elems, c_half_bytes) */
        a_off_bytes = rocke_b_mul(b, a_off_elems, ctx->c_half_bytes);
        /* safe_off = b.select(valid, a_off_bytes, oob_sentinel) */
        safe_off = rocke_b_select(b, valid, a_off_bytes, ctx->oob_sentinel);
        /* a_vec = _buf_load_vN(a_rsrc, safe_off, c0, 2) -- dtype-dispatched */
        if(is_bf16)
            a_vec = rocke_b_buffer_load_vN_bf16(b, ctx->a_rsrc, safe_off, ctx->c0, 2);
        else
            a_vec = rocke_b_buffer_load_vN_f16(b, ctx->a_rsrc, safe_off, ctx->c0, 2);
        /* a_vec = b.select(valid, a_vec, io_vec4_zero) */
        a_vec = rocke_b_select(b, valid, a_vec, ctx->io_vec4_zero);
        /* lds_idx = b.mul(cm["chunk_idx"], b.const_i32(4)) */
        lds_idx = rocke_b_mul(b, ctx->chunk_meta[i].chunk_idx, rocke_b_const_i32(b, 4));

        /* out.append((a_vec, lds_idx)) */
        out_vecs[count] = a_vec;
        out_lds_idx[count] = lds_idx;
        ++count;
    }

    return count;
}

/* ===================================================================== *
 *  Closure: store_to_lds(loads, lds)               (Python lines 564-566)
 * ===================================================================== */
void rocke_dconv16c_store_to_lds(rocke_dconv_16c_ctx_t* ctx,
                                 rocke_value_t* const* vecs,
                                 rocke_value_t* const* lds_idx,
                                 int n,
                                 rocke_value_t* lds)
{
    rocke_ir_builder_t* b = ctx->b;
    int i;

    /* for a_vec, lds_idx in loads:
     *     b.smem_store_vN(lds, [c0, lds_idx], a_vec, 4) -- generic, dtype from value */
    for(i = 0; i < n; ++i)
    {
        rocke_value_t* indices[2];
        indices[0] = ctx->c0;
        indices[1] = lds_idx[i];
        rocke_b_smem_store_vN(b, lds, indices, 2, vecs[i], 4);
    }
}

/* ===================================================================== *
 *  Closure: lds_read_input(q_subtile, s_const, lds) (Python lines 570-590)
 *
 *  Per-lane <4 x half> read from LDS for the s-th column of the 3-wide input
 *  row. The LDS row is laid out flat across (W, G, C) so the W_lds stride is
 *  `BG * cpg` halves.
 * ===================================================================== */
rocke_value_t* rocke_dconv16c_lds_read_input(rocke_dconv_16c_ctx_t* ctx,
                                             int q_subtile,
                                             int s_const,
                                             rocke_value_t* lds)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* W_lds_idx;
    rocke_value_t* lds_idx;
    rocke_value_t* indices[2];

    /* W_lds_idx = b.add(
     *     b.mul(b.add(q_in_lane, b.const_i32(q_subtile*16)), b.const_i32(c_stride)),
     *     b.const_i32(s_const))
     * Force Python left-to-right SSA order: inner add first, then mul, then outer add. */
    {
        rocke_value_t* base = rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 16));
        rocke_value_t* c_s = rocke_b_const_i32(b, ctx->p.stride);
        rocke_value_t* mul_v = rocke_b_mul(b, base, c_s);
        rocke_value_t* c_sc = rocke_b_const_i32(b, s_const);
        W_lds_idx = rocke_b_add(b, mul_v, c_sc);
    }
    /* lds_idx = b.add(b.add(b.mul(W_lds_idx, c_BG_cpg), b.mul(wave_id, c_cpg)),
     *                 b.mul(c4, b.const_i32(4)))
     * Force Python left-to-right SSA emission (C arg eval order unspecified):
     * mul(W_lds_idx,..), mul(wave_id,..), inner add, mul(c4,4), outer add. */
    {
        rocke_value_t* mul_wlds = rocke_b_mul(b, W_lds_idx, ctx->c_BG_cpg);
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_cpg);
        rocke_value_t* inner = rocke_b_add(b, mul_wlds, mul_wave);
        rocke_value_t* mul_c4 = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, 4));
        lds_idx = rocke_b_add(b, inner, mul_c4);
    }
    /* return b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=4) */
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    return rocke_b_smem_load_vN(b, lds, indices, 2, ctx->io_type, 4);
}

/* ===================================================================== *
 *  Closure: lds_read_input_k32(q_subtile, lds)     (Python lines 592-607)
 *
 *  Per-lane <8 x half> read for the folded K=32 MFMA. The lane's c4 selects
 *  S=0/1 (s_lane_k32) and channel block 0/8 (ch_lane_k32).
 * ===================================================================== */
rocke_value_t*
    rocke_dconv16c_lds_read_input_k32(rocke_dconv_16c_ctx_t* ctx, int q_subtile, rocke_value_t* lds)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* W_lds_idx;
    rocke_value_t* lds_idx;
    rocke_value_t* indices[2];

    /* W_lds_idx = b.add(
     *     b.mul(b.add(q_in_lane, b.const_i32(q_subtile*16)), b.const_i32(c_stride)),
     *     s_lane_k32)
     * Force Python left-to-right SSA order: inner add first, then mul, then outer add. */
    {
        rocke_value_t* base = rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 16));
        W_lds_idx = rocke_b_add(
            b, rocke_b_mul(b, base, rocke_b_const_i32(b, ctx->p.stride)), ctx->s_lane_k32);
    }
    /* lds_idx = b.add(b.add(b.mul(W_lds_idx, c_BG_cpg), b.mul(wave_id, c_cpg)),
     *                 ch_lane_k32)
     * Force Python left-to-right SSA emission (C arg eval order unspecified). */
    {
        rocke_value_t* mul_wlds = rocke_b_mul(b, W_lds_idx, ctx->c_BG_cpg);
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_cpg);
        rocke_value_t* inner = rocke_b_add(b, mul_wlds, mul_wave);
        lds_idx = rocke_b_add(b, inner, ctx->ch_lane_k32);
    }
    /* return b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=8) */
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    return rocke_b_smem_load_vN(b, lds, indices, 2, ctx->io_type, 8);
}

/* ===================================================================== *
 *  Closure: lds_read_input_s2_k32(q_subtile, lds)
 *
 *  Per-lane <8 x half> input read for the S=2 residual, promoted to a
 *  zero-padded K=32 atom. Low half (c4 in {0,1}) reads the S=2 column
 *  (W_lds = q_in_lane + 2) at channel block ch_lane_k32; high half (c4 in
 *  {2,3}) is zeroed via select(lane_in_lo_half, vec, fp16x8_zero).
 * ===================================================================== */
rocke_value_t* rocke_dconv16c_lds_read_input_s2_k32(rocke_dconv_16c_ctx_t* ctx,
                                                    int q_subtile,
                                                    rocke_value_t* lds)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* W_lds_idx;
    rocke_value_t* lds_idx;
    rocke_value_t* vec;
    rocke_value_t* indices[2];

    /* W_lds_idx = b.add(
     *     b.mul(b.add(q_in_lane, b.const_i32(q_subtile*16)), b.const_i32(c_stride)),
     *     b.const_i32(2))
     * Force Python left-to-right SSA order: inner add first, then mul, then outer add. */
    {
        rocke_value_t* base = rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 16));
        rocke_value_t* c_s = rocke_b_const_i32(b, ctx->p.stride);
        rocke_value_t* mul_v = rocke_b_mul(b, base, c_s);
        rocke_value_t* c_2 = rocke_b_const_i32(b, 2);
        W_lds_idx = rocke_b_add(b, mul_v, c_2);
    }
    /* lds_idx = b.add(b.add(b.mul(W_lds_idx, c_BG_cpg), b.mul(wave_id, c_cpg)),
     *                 ch_lane_k32) -- force Python left-to-right SSA order. */
    {
        rocke_value_t* mul_wlds = rocke_b_mul(b, W_lds_idx, ctx->c_BG_cpg);
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_cpg);
        rocke_value_t* inner = rocke_b_add(b, mul_wlds, mul_wave);
        lds_idx = rocke_b_add(b, inner, ctx->ch_lane_k32);
    }
    /* vec = b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=8) */
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    vec = rocke_b_smem_load_vN(b, lds, indices, 2, ctx->io_type, 8);
    /* return b.select(lane_in_lo_half, vec, fp16x8_zero) */
    return rocke_b_select(b, ctx->lane_in_lo_half, vec, ctx->fp16x8_zero);
}

/* ===================================================================== *
 *  Prologue prefetch                               (Python lines 609-616)
 *
 *  store_to_lds(issue_dram_load(c0), A_smem); b.sync().
 *  Row 0 = -PAD = -1 is above the image; the descriptor embed flips validity to
 *  false so the loader zero-fills A_smem for iter 0.
 * ===================================================================== */
void rocke_dconv16c_prologue_prefetch(rocke_dconv_16c_ctx_t* ctx)
{
    rocke_value_t* vecs[ROCKE_DCONV16C_MAX_PASSES];
    rocke_value_t* lds_idx[ROCKE_DCONV16C_MAX_PASSES];
    int n;

    /* store_to_lds(issue_dram_load(c0), A_smem) */
    n = rocke_dconv16c_issue_dram_load(ctx, ctx->c0, vecs, lds_idx, ROCKE_DCONV16C_MAX_PASSES);
    if(n < 0)
    {
        return;
    }
    rocke_dconv16c_store_to_lds(ctx, vecs, lds_idx, n, ctx->A_smem);
    /* b.sync() */
    rocke_b_sync(ctx->b);
}

/* ===================================================================== *
 *  The unrolled H-row streaming loop               (Python lines 618-740)
 *
 *  For each of n_iters rows: pick the cur/nxt ping-pong buffer, read inputs from
 *  cur (fold_k32 or per-s), issue next-row DRAM loads, run the per-(qt,r[,s])
 *  MFMA chain into the circular acc slot, store next-row loads to nxt, sync,
 *  then conditionally flush the oldest slot to D and unconditionally reset it.
 * ===================================================================== */
rocke_kernel_def_t* rocke_dconv16c_stream_h_loop(rocke_dconv_16c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int is_bf16 = ctx->p.dtype && strcmp(ctx->p.dtype, "bf16") == 0;
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int KH = p->KH;
    int KW = p->KW;
    int q_subtiles = ctx->q_subtiles;
    int num_accs = q_subtiles * KH;
    int j;
    int qt;

    /* AOT: Hi is a kernarg, so the row count is runtime. Two things in the
     * body are build-time-periodic and would break under a runtime induction
     * variable -- the LDS ping-pong (period 2, and an LDS allocation cannot
     * be selected by a runtime index) and the accumulator slot rotation
     * (period KH). Unrolling the body lcm(2, KH) times aligns both periods
     * with the body boundary: the step is a multiple of KH, so the loop
     * variable is always 0 mod KH at the top and every buffer choice and
     * slot index inside reduces to a constant again.
     *
     * The trip count rounds up, so the last body may run past Hi + KH - 2.
     * That costs nothing and needs no guard: the A descriptor's 0 <= h < Hi
     * bound zero-fills those loads, and the flush predicate drops any row
     * outside [0, Hi). */
    int UNROLL = rocke_dconv_row_loop_unroll(KH, /*lds_ping_pong=*/true);
    rocke_value_t* c_unroll = rocke_b_const_i32(b, UNROLL);
    rocke_value_t* n_iters_v = rocke_b_add(b, ctx->params.p_Hi, rocke_b_const_i32(b, KH - 1));

    /* Output descriptor D[N, Ho, Wo, total_k], built ONCE here (Python builds
     * it at this exact point) so each iteration only pays one offset
     * emission. */
    {
        rocke_dynamic_tensor_descriptor_t* d_dyn
            = rocke_dconv_d_descriptor_dynamic(b, &ctx->params);
        if(!d_dyn)
            return NULL;
        ctx->d_desc = &d_dyn->base;
    }

    rocke_iter_arg_t iter_args[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS];
    char acc_names[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS][32];
    rocke_value_t* accs_flat[ROCKE_DCONV_MAX_QTILES * ROCKE_DCONV_MAX_ACC_SLOTS];
    rocke_for_t for_op;
    rocke_value_t* y_base;

    if(num_accs > (int)(sizeof(iter_args) / sizeof(iter_args[0])))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE; /* too many accumulator slots */
        return NULL;
    }

    /* acc_args = [(f"acc_q{qt}_p{pi}", zero_acc) for qt ... for pi ...] */
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
        rocke_value_t* in_k32[ROCKE_DCONV_MAX_QTILES];
        rocke_value_t* in_s2[ROCKE_DCONV_MAX_QTILES];
        rocke_value_t* in_s[ROCKE_DCONV_MAX_QTILES][16];
        rocke_value_t* loads_next_vecs[ROCKE_DCONV16C_MAX_PASSES];
        rocke_value_t* loads_next_lds[ROCKE_DCONV16C_MAX_PASSES];
        int n_loads_next;
        int P_FLUSH;
        rocke_value_t* p_flush_v;
        rocke_value_t* row_ok;
        rocke_value_t* ho_row_v;

        /* cur = A_smem if (j % 2 == 0 or not double_buffer) else B_smem */
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

        /* Read inputs from the current buffer first; no writes to `cur` are
         * issued until the next time it becomes `nxt`. */
        if(ctx->spec->fold_k32)
        {
            for(qt = 0; qt < q_subtiles; ++qt)
            {
                in_k32[qt] = rocke_dconv16c_lds_read_input_k32(ctx, qt, cur);
                in_s2[qt] = rocke_dconv16c_lds_read_input_s2_k32(ctx, qt, cur);
            }
        }
        else
        {
            for(qt = 0; qt < q_subtiles; ++qt)
            {
                int s_const;
                for(s_const = 0; s_const < KW; ++s_const)
                    in_s[qt][s_const] = rocke_dconv16c_lds_read_input(ctx, qt, s_const, cur);
            }
        }

        /* Issue DRAM reads for the next row into registers before the MFMAs
         * and commit them after, so the VMEM latency overlaps this row's
         * compute. Rows past the image zero-fill through the descriptor
         * bound, so this is unconditional. */
        n_loads_next
            = rocke_dconv16c_issue_dram_load(ctx,
                                             rocke_b_add(b, y_base, rocke_b_const_i32(b, j + 1)),
                                             loads_next_vecs,
                                             loads_next_lds,
                                             ROCKE_DCONV16C_MAX_PASSES);
        if(n_loads_next < 0)
            return NULL;

        for(qt = 0; qt < q_subtiles; ++qt)
        {
            int r_const;
            for(r_const = 0; r_const < KH; ++r_const)
            {
                /* y_base is 0 mod KH (the step is a multiple of KH), so the
                 * rotating slot index collapses to a constant. */
                int p_idx = (((j - r_const) % KH) + KH) % KH;
                int flat = qt * KH + p_idx;
                rocke_value_t* acc_in = accs_flat[flat];

                if(ctx->spec->fold_k32)
                {
                    /* CORRECTNESS-CRITICAL: both folded MFMAs are the SAME
                     * width (16x16x32). S=0/1 fold into one wide atom; S=2 is
                     * promoted to a SECOND wide atom with its upper 16 K
                     * zero-padded. Mixing a 16x16x16 residual into the same
                     * accumulator is a read-after-write accumulator hazard
                     * that both comgr and hipcc miscompile here. */
                    if(is_bf16)
                    {
                        acc_in = rocke_b_mfma_f32_16x16x32_bf16(
                            b, ctx->weights_k32[r_const], in_k32[qt], acc_in);
                        acc_in = rocke_b_mfma_f32_16x16x32_bf16(
                            b, ctx->weights_s2_k32[r_const], in_s2[qt], acc_in);
                    }
                    else
                    {
                        acc_in = rocke_b_mfma_f32_16x16x32_f16(
                            b, ctx->weights_k32[r_const], in_k32[qt], acc_in);
                        acc_in = rocke_b_mfma_f32_16x16x32_f16(
                            b, ctx->weights_s2_k32[r_const], in_s2[qt], acc_in);
                    }
                }
                else
                {
                    int s_const;
                    for(s_const = 0; s_const < KW; ++s_const)
                    {
                        int w_idx = r_const * KW + s_const;
                        if(is_bf16)
                            acc_in = rocke_b_mfma_f32_16x16x16_bf16(
                                b, ctx->weights[w_idx], in_s[qt][s_const], acc_in);
                        else
                            acc_in = rocke_b_mfma_f32_16x16x16_f16(
                                b, ctx->weights[w_idx], in_s[qt][s_const], acc_in);
                    }
                }
                accs_flat[flat] = acc_in;
            }
        }

        /* Single-buffer correctness barrier: with double_buffer False, `cur`
         * and `nxt` are the SAME allocation, so the store below overwrites
         * the row this iteration just read. The next-row DRAM loads are
         * already in registers, so this only orders the LDS access. */
        if(!ctx->spec->double_buffer)
            rocke_b_sync(b);
        rocke_dconv16c_store_to_lds(ctx, loads_next_vecs, loads_next_lds, n_loads_next, nxt);
        rocke_b_sync(b);

        /* Flush the slot completed by this row, then ALWAYS reset it. The
         * unconditional reset is load-bearing: the first KH-1 iterations have
         * a negative flush row and would otherwise leak their r = KH-1
         * contributions into the slot a later, real output row flushes. */
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
        /* Clamp the row index so an out-of-range flush cannot build a wild
         * descriptor offset before the store predicate drops it. */
        ho_row_v = rocke_b_select(b, row_ok, ho_row_v, ctx->c0);

        for(qt = 0; qt < q_subtiles; ++qt)
        {
            int flat = qt * KH + P_FLUSH;
            rocke_value_t* acc_to_flush = accs_flat[flat];
            rocke_value_t* out_q;
            rocke_value_t* store_ok;
            rocke_value_t* k_val;
            rocke_value_t* d_base;
            rocke_value_t* d_valid;
            rocke_value_t* d_base_bytes;
            rocke_value_t* safe_d_off;
            rocke_value_t* acc_h;
            const char* off_names[4];
            rocke_value_t* off_vals[4];

            /* out_q = b.add(b.add(q_tile_start, const_i32(qt*16)), q_in_lane) */
            {
                rocke_value_t* inner
                    = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, qt * 16));
                out_q = rocke_b_add(b, inner, ctx->q_in_lane);
            }
            /* store_ok = b.land(row_ok, b.cmp_lt(out_q, c_W)) */
            store_ok = rocke_b_land(b, row_ok, rocke_b_cmp_lt(b, out_q, ctx->c_W));
            /* k_val = b.add(b.mul(g, c_kpg), b.mul(c4, b.const_i32(4))) --
             * force Python left-to-right SSA emission. */
            {
                rocke_value_t* mul_g = rocke_b_mul(b, ctx->g, ctx->c_kpg);
                rocke_value_t* mul_c4 = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, 4));
                k_val = rocke_b_add(b, mul_g, mul_c4);
            }

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
            /* The 4 per-lane output elements are contiguous in NHWK, so one
             * 64-bit vector store replaces four scalar stores. */
            if(is_bf16)
            {
                acc_h = rocke_b_vec_trunc_f32_to_bf16(b, acc_to_flush);
                rocke_b_buffer_store_vN_bf16(b, ctx->d_rsrc, safe_d_off, ctx->c0, acc_h, 2);
            }
            else
            {
                acc_h = rocke_b_vec_trunc_f32_to_f16(b, acc_to_flush);
                rocke_b_buffer_store_vN_f16(b, ctx->d_rsrc, safe_d_off, ctx->c0, acc_h, 2);
            }
            accs_flat[flat] = ctx->zero_acc;
        }
    }
    rocke_b_scf_yield(b, accs_flat, num_accs);
    rocke_b_region_leave(b);

    return rocke_ir_builder_kernel(b);
}
