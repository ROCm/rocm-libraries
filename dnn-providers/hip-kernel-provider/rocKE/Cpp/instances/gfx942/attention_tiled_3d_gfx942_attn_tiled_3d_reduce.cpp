// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_3d_gfx942_attn_tiled_3d_reduce.c --
 * C99 port of the arch-neutral REDUCE kernel of
 * ck_dsl/instances/gfx942/attention_tiled_3d.py:
 *   build_unified_attention_reduce_tiled   lines 1002-1133
 *
 * Implements the reduce-kernel phase functions declared in
 * ckc/instance_gfx942_attention_tiled_3d_internal.h:
 *   ckc_gfx942_attn_tiled_3d_reduce_config_from_spec   (assist usage)
 *   ckc_gfx942_attention_tiled_3d_reduce_declare_and_prologue (1022-1058)
 *   ckc_gfx942_attention_tiled_3d_reduce_max_pass             (1060-1083)
 *   ckc_gfx942_attention_tiled_3d_reduce_combine_pass         (1085-1104)
 *   ckc_gfx942_attention_tiled_3d_reduce_normalize_pass       (1106-1131)
 *
 * Pure f32 load / exp2 / store; no MFMA, no async DMA. wave64_reduce_max /
 * wave64_reduce_sum come from helper_helper_ck_dsl.helpers.attention.h. The
 * builder-call sequence is a byte-identical translation of the Python span
 * (which is itself a byte-for-byte port of the gfx950 reduce).
 *
 * Peers (segment-kernel phases, ctx init, config_from_spec) live in sibling
 * translation units and are called only via the internal header. This file
 * edits no headers and implements only its scope.
 */

#include <math.h> /* INFINITY */
#include <stdio.h> /* snprintf  */
#include <string.h>

#include "ckc/instance_gfx942_attention_tiled_3d_internal.h"

/* ============================================================ *
 * Per-slot caches shared between reduce_max_pass and reduce_combine_pass.
 *
 * The internal header notes the body .c owns the per-slot caches; ctx only
 * exposes the cross-pass overall_max / overall_expsum / inv_l. The driver runs
 * the passes once, in order, within this single translation unit, so file-scope
 * pointers (arena-allocated, sized by cfg.SEG_PER_LANE) carry the Python
 * seg_idx_of / seg_max_cache / seg_l_cache lists between the two passes.
 *
 * Python tuple (sv, in_rng, sv_safe): in_rng == NULL encodes the Python `None`
 * (the whole lane slot is in range, no guard needed).
 * ============================================================ */
static ckc_value_t** s_seg_sv = NULL; /* sv      per lane slot              */
static ckc_value_t** s_seg_in_rng = NULL; /* in_rng  per lane slot (NULL=None)  */
static ckc_value_t** s_seg_sv_safe = NULL; /* sv_safe per lane slot              */
static ckc_value_t** s_seg_max = NULL; /* seg_max_cache per lane slot        */
static ckc_value_t** s_seg_l = NULL; /* seg_l_cache   per lane slot        */
static int s_seg_slots = 0; /* SEG_PER_LANE captured by pass 1    */

/* ============================================================ *
 * Local descriptor offset helpers (Python `idx, _ = desc.offset(...)` drops
 * the validity). Return NULL on a sticky builder error (no-op propagation).
 * ============================================================ */

static ckc_value_t* ckc__red_ml_offset(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                       ckc_value_t* token,
                                       ckc_value_t* head,
                                       ckc_value_t* seg)
{
    const char* in_names[3] = {"token", "head", "seg"};
    ckc_value_t* in_values[3];
    ckc_value_t* off = NULL;
    ckc_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = seg;
    if(!ckc_transforms_descriptor_offset(
           ctx->b, ctx->ml_desc_red, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

static ckc_value_t* ckc__red_seg_acc_offset(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                            ckc_value_t* token,
                                            ckc_value_t* head,
                                            ckc_value_t* seg,
                                            ckc_value_t* dim)
{
    const char* in_names[4] = {"token", "head", "seg", "dim"};
    ckc_value_t* in_values[4];
    ckc_value_t* off = NULL;
    ckc_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = seg;
    in_values[3] = dim;
    if(!ckc_transforms_descriptor_offset(
           ctx->b, ctx->seg_acc_desc_red, in_names, in_values, 4, &off, &valid))
    {
        return NULL;
    }
    return off;
}

static ckc_value_t* ckc__red_out_offset(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                        ckc_value_t* token,
                                        ckc_value_t* head,
                                        ckc_value_t* dim)
{
    const char* in_names[3] = {"token", "head", "dim"};
    ckc_value_t* in_values[3];
    ckc_value_t* off = NULL;
    ckc_value_t* valid = NULL;
    in_values[0] = token;
    in_values[1] = head;
    in_values[2] = dim;
    if(!ckc_transforms_descriptor_offset(
           ctx->b, ctx->out_desc_red, in_names, in_values, 3, &off, &valid))
    {
        return NULL;
    }
    return off;
}

/* ============================================================ *
 * reduce_config_from_spec assist usage (Python lines 1008-1015).
 *
 * Derives the reduce-kernel compile-time config from the reduce spec and writes
 * the kernel attrs (max_workgroup_size, optional waves_per_eu) the Python body
 * sets at lines 1018-1020. The heavy lifting (HD/NUM_SEG/NUM_QH/dtype/THREADS/
 * HALFS_PER_THREAD/SEG_PER_LANE + the HALFS_PER_THREAD*THREADS == HD assert) is
 * done by the peer ckc_gfx942_attn_tiled_3d_reduce_config_from_spec; this just
 * threads it into the ctx and stamps the kernel attrs.
 * ============================================================ */
void ckc_gfx942_attention_tiled_3d_reduce_declare_and_prologue(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_unified_attention_reduce_tiled_spec_t* spec = ctx->reduce_spec;
    const ckc_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    ckc_param_opts_t opts;

    int NUM_QH = cfg->NUM_QH;
    int NUM_SEG = cfg->NUM_SEG;
    int HD = cfg->HD;

    ckc_value_t* zero_i32;

    /* ---- kernel attrs (lines 1018-1020) ---- */
    if(b != NULL && b->kernel != NULL)
    {
        ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", cfg->THREADS);
        if(spec != NULL && spec->has_waves_per_eu)
        {
            ckc_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
        }
    }

    /* ---- params (lines 1022-1030, load-bearing ABI order) ---- */

    /* output_ptr : ptr<dtype, global>, noalias, writeonly, align=16 */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    ctx->out = ckc_b_param(b, "output_ptr", ckc_ptr_type(b, ctx->cfg.dtype, "global"), &opts);

    /* segm_output_ptr : ptr<f32, global>, readonly, align=16 */
    memset(&opts, 0, sizeof(opts));
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    ctx->seg_out = ckc_b_param(b, "segm_output_ptr", ckc_ptr_type(b, ckc_f32(), "global"), &opts);

    /* segm_max_ptr : ptr<f32, global>, readonly, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->seg_max = ckc_b_param(b, "segm_max_ptr", ckc_ptr_type(b, ckc_f32(), "global"), &opts);

    /* segm_expsum_ptr : ptr<f32, global>, readonly, align=4 */
    ctx->seg_l = ckc_b_param(b, "segm_expsum_ptr", ckc_ptr_type(b, ckc_f32(), "global"), &opts);

    /* seq_lens_ptr : ptr<i32, global>, readonly, align=4 (unused in body) */
    ctx->red_seq_lens = ckc_b_param(b, "seq_lens_ptr", ckc_ptr_type(b, ckc_i32(), "global"), &opts);

    /* ---- grid ids + thread (lines 1032-1034) ---- */
    ctx->q_token = ckc_b_block_id_x(b);
    ctx->q_head = ckc_b_block_id_y(b);
    ctx->tid = ckc_b_thread_id_x(b);

    /* ---- SSA constants (lines 1036-1037) ---- */
    ctx->neg_inf = ckc_b_const_f32(b, -INFINITY);
    ctx->zero_f = ckc_b_const_f32(b, 0.0);

    /* ---- descriptors (lines 1039-1053) ---- */
    {
        int ml_lengths[3] = {1 << 30, NUM_QH, NUM_SEG};
        const char* ml_coords[3] = {"token", "head", "seg"};
        ctx->ml_desc_red
            = ckc_tensor_descriptor_naive(b, "segm_ml", ml_lengths, 3, NULL, ml_coords, 3);
    }
    {
        int sa_lengths[4] = {1 << 30, NUM_QH, NUM_SEG, HD};
        const char* sa_coords[4] = {"token", "head", "seg", "dim"};
        ctx->seg_acc_desc_red
            = ckc_tensor_descriptor_naive(b, "segm_output", sa_lengths, 4, NULL, sa_coords, 4);
    }
    {
        int out_lengths[3] = {1 << 30, NUM_QH, HD};
        const char* out_coords[3] = {"token", "head", "dim"};
        ctx->out_desc_red
            = ckc_tensor_descriptor_naive(b, "out", out_lengths, 3, NULL, out_coords, 3);
    }

    /* ---- base_ml = ml_desc_red.offset(token=q_token, head=q_head, seg=0) ----
     * (line 1055) */
    zero_i32 = ckc_b_const_i32(b, 0);
    ctx->base_ml = ckc__red_ml_offset(ctx, ctx->q_token, ctx->q_head, zero_i32);

    /* ---- factor_lds = smem_alloc_f32([NUM_SEG]) (line 1058) ---- */
    {
        int factor_shape[1] = {NUM_SEG};
        ctx->factor_lds = ckc_b_smem_alloc_f32(b, factor_shape, 1, "seg_factor");
    }
}

/* ============================================================ *
 * pass 1: per-lane partial max over the owned segments (lines 1060-1083).
 *
 *   local_max = neg_inf
 *   for j in range(SEG_PER_LANE):
 *       sv = add(const_i32(j*THREADS), tid)
 *       in_rng = None if (j*THREADS+THREADS) <= NUM_SEG
 *                else cmp_lt(sv, const_i32(NUM_SEG))
 *       sv_safe = sv if in_rng is None else select(in_rng, sv, const_i32(0))
 *       idx = add(base_ml, sv_safe)
 *       ms = global_load_f32(seg_max, idx)
 *       ls = global_load_f32(seg_l,   idx)
 *       if in_rng is not None:
 *           ms = select(in_rng, ms, neg_inf)
 *           ls = select(in_rng, ls, zero_f)
 *       cache (sv, in_rng, sv_safe), ms, ls
 *       local_max = fmax(local_max, ms)
 *   overall_max = _wave64_reduce_max(local_max)
 * ============================================================ */
void ckc_gfx942_attention_tiled_3d_reduce_max_pass(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    int THREADS = cfg->THREADS;
    int NUM_SEG = cfg->NUM_SEG;
    int SEG_PER_LANE = cfg->SEG_PER_LANE;
    ckc_value_t* local_max;
    int j;

    /* Allocate the cross-pass per-slot caches (arena-owned). */
    s_seg_slots = SEG_PER_LANE;
    if(b != NULL && SEG_PER_LANE > 0)
    {
        size_t n = (size_t)SEG_PER_LANE * sizeof(ckc_value_t*);
        s_seg_sv = (ckc_value_t**)ckc_arena_calloc(&b->arena, n);
        s_seg_in_rng = (ckc_value_t**)ckc_arena_calloc(&b->arena, n);
        s_seg_sv_safe = (ckc_value_t**)ckc_arena_calloc(&b->arena, n);
        s_seg_max = (ckc_value_t**)ckc_arena_calloc(&b->arena, n);
        s_seg_l = (ckc_value_t**)ckc_arena_calloc(&b->arena, n);
    }

    local_max = ctx->neg_inf;

    for(j = 0; j < SEG_PER_LANE; ++j)
    {
        ckc_value_t* sv;
        ckc_value_t* in_rng;
        ckc_value_t* sv_safe;
        ckc_value_t* idx;
        ckc_value_t* ms;
        ckc_value_t* ls;

        sv = ckc_b_add(b, ckc_b_const_i32(b, (int64_t)j * THREADS), ctx->tid);

        if((j * THREADS + THREADS) <= NUM_SEG)
        {
            in_rng = NULL;
        }
        else
        {
            in_rng = ckc_b_cmp_lt(b, sv, ckc_b_const_i32(b, NUM_SEG));
        }

        if(in_rng == NULL)
        {
            sv_safe = sv;
        }
        else
        {
            sv_safe = ckc_b_select(b, in_rng, sv, ckc_b_const_i32(b, 0));
        }

        idx = ckc_b_add(b, ctx->base_ml, sv_safe);
        ms = ckc_b_global_load_f32(b, ctx->seg_max, idx, 0);
        ls = ckc_b_global_load_f32(b, ctx->seg_l, idx, 0);

        if(in_rng != NULL)
        {
            ms = ckc_b_select(b, in_rng, ms, ctx->neg_inf);
            ls = ckc_b_select(b, in_rng, ls, ctx->zero_f);
        }

        if(s_seg_sv != NULL)
        {
            s_seg_sv[j] = sv;
            s_seg_in_rng[j] = in_rng;
            s_seg_sv_safe[j] = sv_safe;
            s_seg_max[j] = ms;
            s_seg_l[j] = ls;
        }

        local_max = ckc_b_fmax(b, local_max, ms);
    }

    ctx->overall_max = ckc_wave64_reduce_max(b, local_max);
}

/* ============================================================ *
 * pass 2: per-lane partial expsum + cache per-segment factor (lines 1085-1104).
 *
 *   local_den = zero_f
 *   for j in range(SEG_PER_LANE):
 *       sv, in_rng, sv_safe = seg_idx_of[j]
 *       ms = seg_max_cache[j]; ls = seg_l_cache[j]
 *       ms_finite = fcmp("ogt", ms, neg_inf)
 *       factor_raw = exp2(fsub(ms, overall_max))
 *       factor = select(ms_finite, factor_raw, zero_f)
 *       local_den = fadd(local_den, fmul(ls, factor))
 *       if in_rng is None:
 *           smem_store_vN_f32(factor_lds, [sv_safe], factor, 1)
 *       else:
 *           with scf_if(in_rng):
 *               smem_store_vN_f32(factor_lds, [sv], factor, 1)
 *   overall_expsum = _wave64_reduce_sum(local_den)
 *   safe_expsum = fcmp("oeq", overall_expsum, zero_f)
 *   inv_l = select(safe_expsum, zero_f, rcp(overall_expsum))
 *   sync()
 * ============================================================ */
void ckc_gfx942_attention_tiled_3d_reduce_combine_pass(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    ckc_value_t* local_den;
    ckc_value_t* safe_expsum;
    int j;

    local_den = ctx->zero_f;

    for(j = 0; j < s_seg_slots; ++j)
    {
        ckc_value_t* sv;
        ckc_value_t* in_rng;
        ckc_value_t* sv_safe;
        ckc_value_t* ms;
        ckc_value_t* ls;
        ckc_value_t* ms_finite;
        ckc_value_t* factor_raw;
        ckc_value_t* factor;

        sv = s_seg_sv ? s_seg_sv[j] : NULL;
        in_rng = s_seg_in_rng ? s_seg_in_rng[j] : NULL;
        sv_safe = s_seg_sv_safe ? s_seg_sv_safe[j] : NULL;
        ms = s_seg_max ? s_seg_max[j] : NULL;
        ls = s_seg_l ? s_seg_l[j] : NULL;

        ms_finite = ckc_b_fcmp(b, "ogt", ms, ctx->neg_inf);
        factor_raw = ckc_b_exp2(b, ckc_b_fsub(b, ms, ctx->overall_max));
        factor = ckc_b_select(b, ms_finite, factor_raw, ctx->zero_f);

        local_den = ckc_b_fadd(b, local_den, ckc_b_fmul(b, ls, factor));

        if(in_rng == NULL)
        {
            ckc_value_t* indices[1];
            indices[0] = sv_safe;
            ckc_b_smem_store_vN_f32(b, ctx->factor_lds, indices, 1, factor, 1);
        }
        else
        {
            ckc_if_t guard = ckc_b_scf_if(b, in_rng);
            ckc_b_region_enter(b, guard.then_region);
            {
                ckc_value_t* indices[1];
                indices[0] = sv;
                ckc_b_smem_store_vN_f32(b, ctx->factor_lds, indices, 1, factor, 1);
            }
            ckc_b_region_leave(b);
        }
    }

    ctx->overall_expsum = ckc_wave64_reduce_sum(b, local_den);
    safe_expsum = ckc_b_fcmp(b, "oeq", ctx->overall_expsum, ctx->zero_f);
    ctx->inv_l = ckc_b_select(b, safe_expsum, ctx->zero_f, ckc_b_rcp(b, ctx->overall_expsum));

    ckc_b_sync(b);
}

/* ============================================================ *
 * pass 3: per-element reduce + normalize + write (lines 1106-1131).
 *
 *   for li in range(HALFS_PER_THREAD):
 *       d = add(mul(const_i32(li), const_i32(THREADS)), tid)
 *       acc_loop = scf_for_iter(0, NUM_SEG, 1, [("ac{li}", zero_f)], iv_name="s_acc{li}")
 *       with acc_loop as (sv, (ac,)):
 *           factor = smem_load_vN_f32(factor_lds, sv, n=1)
 *           factor_s = vec_extract(factor, 0)
 *           idx_acc, _ = seg_acc_desc_red.offset(token=q_token, head=q_head, seg=sv, dim=d)
 *           ov = global_load_f32(seg_out, idx_acc)
 *           scf_yield(fadd(ac, fmul(ov, factor_s)))
 *       scalar_out_f32 = fmul(acc_loop.results[0], inv_l)
 *       scalar_out = cast_f32_to(scalar_out_f32, dtype)
 *       out_idx, _ = out_desc_red.offset(token=q_token, head=q_head, dim=d)
 *       global_store(out, out_idx, scalar_out, align=2)
 * ============================================================ */
void ckc_gfx942_attention_tiled_3d_reduce_normalize_pass(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_gfx942_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    int THREADS = cfg->THREADS;
    int NUM_SEG = cfg->NUM_SEG;
    int HALFS_PER_THREAD = cfg->HALFS_PER_THREAD;
    int li;

    for(li = 0; li < HALFS_PER_THREAD; ++li)
    {
        ckc_value_t* d;
        ckc_for_t acc_loop;
        ckc_iter_arg_t iter_args[1];
        char acname[32];
        char ivname[32];
        ckc_value_t* scalar_out_f32;
        ckc_value_t* scalar_out;
        ckc_value_t* out_idx;

        {
            ckc_value_t* d_li = ckc_b_const_i32(b, li);
            ckc_value_t* d_thr = ckc_b_const_i32(b, THREADS);
            d = ckc_b_add(b, ckc_b_mul(b, d_li, d_thr), ctx->tid);
        }

        snprintf(acname, sizeof(acname), "ac%d", li);
        snprintf(ivname, sizeof(ivname), "s_acc%d", li);

        iter_args[0].name = acname;
        iter_args[0].init = ctx->zero_f;

        acc_loop = ckc_b_scf_for_iter(b,
                                      ckc_b_const_i32(b, 0),
                                      ckc_b_const_i32(b, NUM_SEG),
                                      ckc_b_const_i32(b, 1),
                                      iter_args,
                                      1,
                                      ivname,
                                      false,
                                      true);

        ckc_b_region_enter(b, acc_loop.body);
        {
            ckc_value_t* sv = acc_loop.iv;
            ckc_value_t* ac = acc_loop.iter_vars ? acc_loop.iter_vars[0] : NULL;
            ckc_value_t* factor;
            ckc_value_t* factor_s;
            ckc_value_t* idx_acc;
            ckc_value_t* ov;
            ckc_value_t* indices[1];
            ckc_value_t* yields[1];

            indices[0] = sv;
            factor = ckc_b_smem_load_vN_f32(b, ctx->factor_lds, indices, 1, 1);
            factor_s = ckc_b_vec_extract(b, factor, 0);

            idx_acc = ckc__red_seg_acc_offset(ctx, ctx->q_token, ctx->q_head, sv, d);
            ov = ckc_b_global_load_f32(b, ctx->seg_out, idx_acc, 0);

            yields[0] = ckc_b_fadd(b, ac, ckc_b_fmul(b, ov, factor_s));
            ckc_b_scf_yield(b, yields, 1);
        }
        ckc_b_region_leave(b);

        scalar_out_f32 = ckc_b_fmul(
            b,
            (acc_loop.op != NULL && acc_loop.op->num_results >= 1) ? acc_loop.op->results[0] : NULL,
            ctx->inv_l);
        scalar_out = ckc_b_cast_f32_to(b, scalar_out_f32, ctx->cfg.dtype);

        out_idx = ckc__red_out_offset(ctx, ctx->q_token, ctx->q_head, d);
        ckc_b_global_store(b, ctx->out, out_idx, scalar_out, 2);
    }
}
