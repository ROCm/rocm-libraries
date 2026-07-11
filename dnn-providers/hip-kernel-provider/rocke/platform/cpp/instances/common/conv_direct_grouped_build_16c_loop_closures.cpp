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
#include "rocke/instance_conv_direct_grouped_internal.h"

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
        /* a_vec = b.buffer_load_vN_f16(a_rsrc, safe_off, c0, 2) */
        a_vec = rocke_b_buffer_load_vN_f16(b, ctx->a_rsrc, safe_off, ctx->c0, 2);
        /* a_vec = b.select(valid, a_vec, fp16x4_zero) */
        a_vec = rocke_b_select(b, valid, a_vec, ctx->fp16x4_zero);
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
     *     b.smem_store_vN_f16(lds, [c0, lds_idx], a_vec, 4) */
    for(i = 0; i < n; ++i)
    {
        rocke_value_t* indices[2];
        indices[0] = ctx->c0;
        indices[1] = lds_idx[i];
        rocke_b_smem_store_vN_f16(b, lds, indices, 2, vecs[i], 4);
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

    /* W_lds_idx = b.add(q_in_lane, b.const_i32(q_subtile * 16 + s_const)) */
    W_lds_idx = rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 16 + s_const));
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
    /* return b.smem_load_vN_f16(lds, c0, lds_idx, n=4) */
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    return rocke_b_smem_load_vN_f16(b, lds, indices, 2, 4);
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

    /* W_lds_idx = b.add(b.add(q_in_lane, b.const_i32(q_subtile * 16)),
     *                   s_lane_k32) */
    W_lds_idx = rocke_b_add(
        b, rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 16)), ctx->s_lane_k32);
    /* lds_idx = b.add(b.add(b.mul(W_lds_idx, c_BG_cpg), b.mul(wave_id, c_cpg)),
     *                 ch_lane_k32)
     * Force Python left-to-right SSA emission (C arg eval order unspecified). */
    {
        rocke_value_t* mul_wlds = rocke_b_mul(b, W_lds_idx, ctx->c_BG_cpg);
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_cpg);
        rocke_value_t* inner = rocke_b_add(b, mul_wlds, mul_wave);
        lds_idx = rocke_b_add(b, inner, ctx->ch_lane_k32);
    }
    /* return b.smem_load_vN_f16(lds, c0, lds_idx, n=8) */
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    return rocke_b_smem_load_vN_f16(b, lds, indices, 2, 8);
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

    /* W_lds_idx = b.add(q_in_lane, b.const_i32(q_subtile*16 + 2)) */
    W_lds_idx = rocke_b_add(b, ctx->q_in_lane, rocke_b_const_i32(b, q_subtile * 16 + 2));
    /* lds_idx = b.add(b.add(b.mul(W_lds_idx, c_BG_cpg), b.mul(wave_id, c_cpg)),
     *                 ch_lane_k32) -- force Python left-to-right SSA order. */
    {
        rocke_value_t* mul_wlds = rocke_b_mul(b, W_lds_idx, ctx->c_BG_cpg);
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_cpg);
        rocke_value_t* inner = rocke_b_add(b, mul_wlds, mul_wave);
        lds_idx = rocke_b_add(b, inner, ctx->ch_lane_k32);
    }
    /* vec = b.smem_load_vN_f16(lds, c0, lds_idx, n=8) */
    indices[0] = ctx->c0;
    indices[1] = lds_idx;
    vec = rocke_b_smem_load_vN_f16(b, lds, indices, 2, 8);
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
    const rocke_direct_conv_problem_t* p = &ctx->p;
    int KH = p->KH;
    int KW = p->KW;
    int y;

    /* n_iters = p.H + p.KH - 1  (already in ctx->n_iters) */
    int n_iters = ctx->n_iters;
    int q_subtiles = ctx->q_subtiles;

    /* acc_tiles seeded to zero_acc (q_subtiles x KH circular slots). Python:
     *   acc_tiles = [[zero_acc, zero_acc, zero_acc] for _ in range(q_subtiles)]
     * Mirror that here; the prologue/descriptor phase may have seeded it, but the
     * Python builds this list at the top of the streaming section. */
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
        /* Per-q inputs. fold_k32: inputs_by_q[qt] = (input_k32, input_s2).
         * else: inputs_by_q[qt][s] for s in range(KW). */
        rocke_value_t* in_k32[ROCKE_DCONV_MAX_QTILES];
        rocke_value_t* in_s2[ROCKE_DCONV_MAX_QTILES];
        rocke_value_t* in_s[ROCKE_DCONV_MAX_QTILES][16];
        rocke_value_t* loads_next_vecs[ROCKE_DCONV16C_MAX_PASSES];
        rocke_value_t* loads_next_lds[ROCKE_DCONV16C_MAX_PASSES];
        int n_loads_next = 0;
        bool has_loads_next = false;
        int qt;
        int p_flush_val;
        int P_FLUSH;

        /* cur = A_smem if (y % 2 == 0 or not double_buffer) else B_smem
         * nxt = B_smem if (y % 2 == 0 or not double_buffer) else A_smem */
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

        /* Read inputs from the current buffer first. */
        if(ctx->spec->fold_k32)
        {
            for(qt = 0; qt < q_subtiles; ++qt)
            {
                /* (lds_read_input_k32(qt, cur), lds_read_input_s2_k32(qt, cur)) */
                in_k32[qt] = rocke_dconv16c_lds_read_input_k32(ctx, qt, cur);
                in_s2[qt] = rocke_dconv16c_lds_read_input_s2_k32(ctx, qt, cur);
            }
        }
        else
        {
            for(qt = 0; qt < q_subtiles; ++qt)
            {
                int s;
                for(s = 0; s < KW; ++s)
                {
                    in_s[qt][s] = rocke_dconv16c_lds_read_input(ctx, qt, s, cur);
                }
            }
        }

        /* loads_next = issue_dram_load(b.const_i32(y + 1)) if y+1 < n_iters */
        if(y + 1 < n_iters)
        {
            n_loads_next = rocke_dconv16c_issue_dram_load(ctx,
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

        /* MFMA chain into the circular acc slot. */
        for(qt = 0; qt < q_subtiles; ++qt)
        {
            int r_const;
            for(r_const = 0; r_const < KH; ++r_const)
            {
                /* p_idx = (y - r_const) % KH */
                int p_idx = (((y - r_const) % KH) + KH) % KH;
                rocke_value_t* acc_in = ctx->acc_tiles[qt][p_idx];

                if(ctx->spec->fold_k32)
                {
                    /* All-wide fold (CORRECTNESS-CRITICAL): both folded MFMAs
                     * are the SAME width (16x16x32). S=0/1 fold into one wide
                     * atom; S=2 is promoted to a SECOND wide atom with its
                     * upper 16 K zero-padded (weights_s2_k32 / in_s2). Chaining
                     * two same-width atoms on one accumulator matches the
                     * mfma_gemm hero path. Mixing a 16x16x16 residual into the
                     * same accumulator as the 16x16x32 atom -- a narrow MFMA
                     * whose C-operand is the just-written result of a wide MFMA
                     * (or vice versa) -- is a read-after-write accumulator
                     * hazard that BOTH comgr and hipcc miscompile in this
                     * fully-unrolled kernel, corrupting H-edge output rows in a
                     * shape-dependent way. Op order matches Python:
                     *   acc_in = mfma_f32_16x16x32_f16(weights_k32[r], in_k32, acc_in)
                     *   acc_in = mfma_f32_16x16x32_f16(weights_s2_k32[r], in_s2, acc_in) */
                    acc_in = rocke_b_mfma_f32_16x16x32_f16(
                        b, ctx->weights_k32[r_const], in_k32[qt], acc_in);
                    acc_in = rocke_b_mfma_f32_16x16x32_f16(
                        b, ctx->weights_s2_k32[r_const], in_s2[qt], acc_in);
                }
                else
                {
                    int s_const;
                    for(s_const = 0; s_const < KW; ++s_const)
                    {
                        /* w_idx = r_const * KW + s_const */
                        int w_idx = r_const * KW + s_const;
                        acc_in = rocke_b_mfma_f32_16x16x16_f16(
                            b, ctx->weights[w_idx], in_s[qt][s_const], acc_in);
                    }
                }
                /* accs[p_idx] = acc_in */
                ctx->acc_tiles[qt][p_idx] = acc_in;
            }
        }

        /* if loads_next is not None: store_to_lds(loads_next, nxt) */
        if(has_loads_next)
        {
            /* Single-buffer correctness barrier. When double_buffer is
             * False, cur and nxt are the SAME LDS allocation, so the
             * store_to_lds below overwrites the row this iteration just read
             * via lds_read_input. With more than one wave per workgroup
             * (block_groups > 1) the only barrier used to be the one at the
             * end of the iteration, so a fast wave could begin storing row
             * y+1 into LDS while a slower wave was still issuing its ds_reads
             * for row y -- a read-after-write race that corrupted the slower
             * waves' inputs (seen as nondeterministic wrong outputs in the
             * interior waves/groups and near the H/W edges). The next-row
             * DRAM loads were already issued into registers above, so this
             * barrier only forces every wave to finish reading the current
             * LDS row before any wave overwrites it; the MFMAs above overlap
             * the ds_read latency. The double-buffer path doesn't need it
             * (the store targets the other ping-pong buffer). */
            if(!ctx->spec->double_buffer)
            {
                rocke_b_sync(b);
            }
            rocke_dconv16c_store_to_lds(ctx, loads_next_vecs, loads_next_lds, n_loads_next, nxt);
        }
        /* b.sync() */
        rocke_b_sync(b);

        /* p_flush_val = y - (KH - 1) ; P_FLUSH = p_flush_val % KH */
        p_flush_val = y - (KH - 1);
        P_FLUSH = ((p_flush_val % KH) + KH) % KH;

        /* if 0 <= p_flush_val < p.H: flush each qt to D */
        if(0 <= p_flush_val && p_flush_val < p->H)
        {
            for(qt = 0; qt < q_subtiles; ++qt)
            {
                rocke_value_t* acc_to_flush = ctx->acc_tiles[qt][P_FLUSH];
                rocke_value_t* out_q;
                rocke_value_t* out_q_valid;
                rocke_value_t* k_val;
                rocke_value_t* d_base;
                rocke_value_t* d_valid;
                rocke_value_t* d_base_bytes;
                rocke_value_t* safe_d_off;
                rocke_value_t* acc_h;
                const char* off_names[4];
                rocke_value_t* off_vals[4];

                /* out_q = b.add(b.add(q_tile_start, b.const_i32(qt * 16)), q_in_lane) */
                out_q
                    = rocke_b_add(b,
                                  rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, qt * 16)),
                                  ctx->q_in_lane);
                /* out_q_valid = b.cmp_lt(out_q, c_W) */
                out_q_valid = rocke_b_cmp_lt(b, out_q, ctx->c_W);
                /* k_val = b.add(b.mul(g, c_kpg), b.mul(c4, b.const_i32(4)))
                 * Force Python left-to-right SSA emission (C arg eval order
                 * is unspecified). */
                {
                    rocke_value_t* mul_g = rocke_b_mul(b, ctx->g, ctx->c_kpg);
                    rocke_value_t* mul_c4 = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, 4));
                    k_val = rocke_b_add(b, mul_g, mul_c4);
                }

                /* d_base, _ = d_desc.offset(b, n=n, h=const_i32(p_flush_val),
                 *                           w=out_q, k=k_val) */
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

                /* d_base_bytes = b.mul(d_base, c_half_bytes) */
                d_base_bytes = rocke_b_mul(b, d_base, ctx->c_half_bytes);
                /* safe_d_off = b.select(out_q_valid, d_base_bytes, oob_sentinel) */
                safe_d_off = rocke_b_select(b, out_q_valid, d_base_bytes, ctx->oob_sentinel);
                /* acc_h = b.vec_trunc_f32_to_f16(acc_to_flush) */
                acc_h = rocke_b_vec_trunc_f32_to_f16(b, acc_to_flush);
                /* b.buffer_store_vN_f16(d_rsrc, safe_d_off, c0, acc_h, 2) */
                rocke_b_buffer_store_vN_f16(b, ctx->d_rsrc, safe_d_off, ctx->c0, acc_h, 2);
            }
        }

        /* Unconditional slot reset: accs[P_FLUSH] = zero_acc for each qt. */
        for(qt = 0; qt < q_subtiles; ++qt)
        {
            ctx->acc_tiles[qt][P_FLUSH] = ctx->zero_acc;
        }
    }

    /* return b.kernel */
    return rocke_ir_builder_kernel(b);
}
