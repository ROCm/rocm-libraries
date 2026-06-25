// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_build_16c_phases.c -- C99 port of the IR-emitting
 * prologue/setup phases of build_direct_conv_16c
 * (rocke/instances/common/conv_direct_grouped.py, lines 256-519 + 637-641).
 *
 * Implements exactly four phase functions over rocke_dconv_16c_ctx_t:
 *   rocke_dconv16c_prologue          (Python 256-355)
 *   rocke_dconv16c_load_weights      (Python 357-415)
 *   rocke_dconv16c_build_chunk_meta  (Python 444-473)
 *   rocke_dconv16c_build_descriptors (Python 475-519, 637-641)
 *
 * Peers (issue_dram_load / store_to_lds / lds_read_input* / prologue_prefetch /
 * stream_h_loop) live in sibling TUs and are reached via the internal header.
 * The builder-call sequence here is byte-identical to the Python source.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  Prologue (Python lines 256-355).
 *
 *  spec.validate() then is_valid_spec_16c gate; derive every geometry scalar;
 *  declare params; build all SSA constants; decode thread/wave/lane +
 *  grid/group; alloc the LDS ping-pong; build buffer rsrcs.
 *
 *  NOTE: unlike the Python (which constructs IRBuilder(spec.kernel_name())
 *  here), the C build entry pre-inits ctx->b with the kernel name and does NOT
 *  re-init it; this phase only sets the max_workgroup_size attr and proceeds.
 * ===================================================================== */
bool rocke_dconv16c_prologue(rocke_dconv_16c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_conv_16c_spec_t* spec = ctx->spec;
    char reason[ROCKE_ERR_MSG_CAP];

    /* spec.validate() -- hard invariant assertions. */
    if(rocke_direct_conv_16c_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE; /* mirror Python AssertionError -> sticky error */
        return false;
    }

    /* ok, why = is_valid_spec_16c(spec, arch=arch); if not ok: raise ValueError */
    if(!rocke_direct_conv_16c_is_valid_spec(spec, ctx->arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return false;
    }

    /* p = spec.problem  (already copied by value into ctx->p by the driver, but
     * mirror the Python local binding explicitly). */
    ctx->p = spec->problem;

    /* ---- block-geometry scalars ---- */
    ctx->BLOCK_Q = spec->block_q;
    ctx->BLOCK_GROUPS = spec->block_groups;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_conv_16c_threads_per_block(spec);
    ctx->LDS_W = ctx->BLOCK_Q + ctx->p.KW - 1;
    ctx->LDS_ROW_FP16 = ctx->LDS_W * ctx->BLOCK_GROUPS * ctx->p.cpg;
    ctx->LOAD_VEC = 4;
    ctx->NUM_VEC4 = ctx->LDS_ROW_FP16 / ctx->LOAD_VEC;

    if(ctx->NUM_VEC4 == 0)
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE; /* "LDS row too small for one vec4 per thread" */
        return false;
    }
    ctx->PASSES = (ctx->NUM_VEC4 + ctx->THREADS - 1) / ctx->THREADS;

    /* derived geometry referenced by later phases / the H-loop */
    ctx->q_subtiles = ctx->BLOCK_Q / 16;
    ctx->n_iters = ctx->p.H + ctx->p.KH - 1;

    /* b.kernel.attrs["max_workgroup_size"] = THREADS */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

    /* ---- params ---- */
    {
        const rocke_type_t* f16ptr = rocke_ptr_type(b, rocke_f16(), "global");
        rocke_param_opts_t ro;
        rocke_param_opts_t wo;
        rocke_param_opts_t none;

        /* A = b.param("A", PtrType(F16,"global"), noalias=True, readonly=True, align=16) */
        ro = (rocke_param_opts_t){0};
        ro.noalias = true;
        ro.noalias_set = true;
        ro.readonly = true;
        ro.readonly_set = true;
        ro.align = 16;
        ro.align_set = true;
        ctx->A = rocke_b_param(b, "A", f16ptr, &ro);
        ctx->Bp = rocke_b_param(b, "B", f16ptr, &ro);

        /* D = b.param("D", PtrType(F16,"global"), noalias=True, writeonly=True, align=16) */
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

    /* ---- common SSA constants ---- */
    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    ctx->c_BG = rocke_b_const_i32(b, ctx->BLOCK_GROUPS);
    ctx->c_BQ = rocke_b_const_i32(b, ctx->BLOCK_Q);
    ctx->c_cpg = rocke_b_const_i32(b, ctx->p.cpg);
    ctx->c_kpg = rocke_b_const_i32(b, ctx->p.kpg);
    ctx->c_W = rocke_b_const_i32(b, ctx->p.W);

    /* c_BG_cpg = b.const_i32(BLOCK_GROUPS * p.cpg) */
    ctx->c_BG_cpg = rocke_b_const_i32(b, ctx->BLOCK_GROUPS * ctx->p.cpg);

    /* ---- thread / wave / lane decode ---- */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);
    ctx->c4 = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, 16)); /* 0..3 */
    ctx->q_in_lane = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, 16)); /* 0..15 */
    ctx->s_lane_k32 = rocke_b_div(b, ctx->c4, rocke_b_const_i32(b, 2));
    /* ch_lane_k32 = b.mul(b.mod(c4, b.const_i32(2)), b.const_i32(8))
     * Python evaluates the mul's first arg (the mod, incl its const_i32(2))
     * fully before its second arg (const_i32(8)). C argument evaluation order
     * is unspecified, so force the Python left-to-right SSA emission with a
     * temp for the inner mod. */
    {
        rocke_value_t* mod_c4_2 = rocke_b_mod(b, ctx->c4, rocke_b_const_i32(b, 2));
        ctx->ch_lane_k32 = rocke_b_mul(b, mod_c4_2, rocke_b_const_i32(b, 8));
    }
    ctx->ch_lane_k16 = rocke_b_mul(b, ctx->c4, rocke_b_const_i32(b, 4));

    /* ---- grid / group decode ---- */
    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->n = rocke_b_block_id_z(b);

    ctx->g_tile = ctx->by;
    ctx->g = rocke_b_add(b, rocke_b_mul(b, ctx->g_tile, ctx->c_BG), ctx->wave_id);
    ctx->q_tile_start = rocke_b_mul(b, ctx->bx, ctx->c_BQ);

    /* ---- LDS ping-pong buffers ---- */
    ctx->lds_total_fp16 = ctx->PASSES * ctx->THREADS * ctx->LOAD_VEC;
    {
        int shape[2];
        shape[0] = 1;
        shape[1] = ctx->lds_total_fp16;
        ctx->A_smem = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "lds_a");
        if(spec->double_buffer)
            ctx->B_smem = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "lds_b");
        else
            ctx->B_smem = ctx->A_smem;
    }

    /* ---- buffer rsrcs ---- */
    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);
    ctx->d_rsrc = rocke_b_buffer_rsrc(b, ctx->D, ctx->D_bytes);

    ctx->c_half_bytes = rocke_b_const_i32(b, 2);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    ctx->fp16x4_zero = rocke_b_zero_vec_f16(b, 4);
    ctx->zero_acc = rocke_b_zero_vec_f32(b, 4);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Weight-load phase (Python lines 357-415).
 *
 *  b_desc = TensorDescriptor.naive("B", [total_k, KH, KW, cpg],
 *                                  coord_names=(k_out, r, s, c))
 *  k_out_val = g*kpg + q_in_lane
 *  fold_k32 -> per-r folded K=32 (<8 x half>) + residual K=16 (<2 dwords>);
 *  else      -> weights[r*KW+s] K=16.
 * ===================================================================== */
void rocke_dconv16c_load_weights(rocke_dconv_16c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);

    /* TensorDescriptor.naive("B", lengths=[total_k, KH, KW, cpg],
     *                        coord_names=("k_out","r","s","c")) */
    {
        int lengths[4];
        static const char* const coord_names[4] = {"k_out", "r", "s", "c"};
        lengths[0] = total_k;
        lengths[1] = ctx->p.KH;
        lengths[2] = ctx->p.KW;
        lengths[3] = ctx->p.cpg;
        ctx->b_desc = rocke_tensor_descriptor_naive(b, "B", lengths, 4, NULL, coord_names, 4);
    }

    /* k_out_val = b.add(b.mul(g, c_kpg), q_in_lane) */
    ctx->k_out_val = rocke_b_add(b, rocke_b_mul(b, ctx->g, ctx->c_kpg), ctx->q_in_lane);

    ctx->n_weights = 0;
    ctx->n_weights_k32 = 0;

    /* lane_in_lo_half = b.cmp_lt(c4, b.const_i32(2))
     * fp16x8_zero     = b.zero_vec_f16(8)
     * (emitted here, matching the Python SSA order, before the per-r loop). */
    ctx->lane_in_lo_half = rocke_b_cmp_lt(b, ctx->c4, rocke_b_const_i32(b, 2));
    ctx->fp16x8_zero = rocke_b_zero_vec_f16(b, 8);

    if(ctx->spec->fold_k32)
    {
        int r_const;
        for(r_const = 0; r_const < ctx->p.KH; ++r_const)
        {
            rocke_value_t* r_i = rocke_b_const_i32(b, r_const);
            rocke_value_t* w_off_k32 = NULL;
            rocke_value_t* w_off_s2 = NULL;
            rocke_value_t* w_s2 = NULL;
            rocke_value_t* valid = NULL;

            /* Fold S=0,S=1 into one K=32 MFMA: <8 x half> at
             *   s_lane_k32*cpg + ch_lane_k32 */
            {
                const char* in_names[4] = {"k_out", "r", "s", "c"};
                rocke_value_t* in_values[4];
                in_values[0] = ctx->k_out_val;
                in_values[1] = r_i;
                in_values[2] = ctx->s_lane_k32;
                in_values[3] = ctx->ch_lane_k32;
                rocke_transforms_descriptor_offset(
                    b, ctx->b_desc, in_names, in_values, 4, &w_off_k32, &valid);
            }
            ctx->weights_k32[r_const] = rocke_b_buffer_load_vN_f16(
                b, ctx->b_rsrc, rocke_b_mul(b, w_off_k32, ctx->c_half_bytes), ctx->c0, 4);

            /* Residual S=2 promoted to a zero-padded K=32 atom: low half
             * (c4 in {0,1}) carries B[k_out,r,2,0:8]/[8:16]; high half zero. */
            {
                const char* in_names[4] = {"k_out", "r", "s", "c"};
                rocke_value_t* in_values[4];
                in_values[0] = ctx->k_out_val;
                in_values[1] = r_i;
                in_values[2] = rocke_b_const_i32(b, 2);
                in_values[3] = ctx->ch_lane_k32;
                rocke_transforms_descriptor_offset(
                    b, ctx->b_desc, in_names, in_values, 4, &w_off_s2, &valid);
            }
            w_s2 = rocke_b_buffer_load_vN_f16(
                b, ctx->b_rsrc, rocke_b_mul(b, w_off_s2, ctx->c_half_bytes), ctx->c0, 4);
            ctx->weights_s2_k32[r_const]
                = rocke_b_select(b, ctx->lane_in_lo_half, w_s2, ctx->fp16x8_zero);
        }
        ctx->n_weights_k32 = ctx->p.KH;
    }
    else
    {
        int r_const, s_const;
        for(r_const = 0; r_const < ctx->p.KH; ++r_const)
        {
            for(s_const = 0; s_const < ctx->p.KW; ++s_const)
            {
                rocke_value_t* r_i = rocke_b_const_i32(b, r_const);
                rocke_value_t* s_i = rocke_b_const_i32(b, s_const);
                rocke_value_t* w_off = NULL;
                rocke_value_t* valid = NULL;
                const char* in_names[4] = {"k_out", "r", "s", "c"};
                rocke_value_t* in_values[4];
                in_values[0] = ctx->k_out_val;
                in_values[1] = r_i;
                in_values[2] = s_i;
                in_values[3] = ctx->ch_lane_k16;
                rocke_transforms_descriptor_offset(
                    b, ctx->b_desc, in_names, in_values, 4, &w_off, &valid);
                ctx->weights[ctx->n_weights++] = rocke_b_buffer_load_vN_f16(
                    b, ctx->b_rsrc, rocke_b_mul(b, w_off, ctx->c_half_bytes), ctx->c0, 2);
            }
        }
    }
}

/* ===================================================================== *
 *  Chunk-decode phase (Python lines 444-473).
 *
 *  chunk_desc = naive("chunk_unmerge", [LDS_W, BLOCK_GROUPS, 4],
 *                     coord_names=(W_lds, group_in_wg, ch_block))
 *               .transform(unmerge_magic("chunk_idx", into=(...), dims=[...]))
 *  for pass_idx in range(PASSES): decode + in_bounds + abs_group.
 * ===================================================================== */
void rocke_dconv16c_build_chunk_meta(rocke_dconv_16c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    static const char* const coord_names[3] = {"W_lds", "group_in_wg", "ch_block"};
    int lengths[3];
    int pass_idx;
    rocke_tensor_descriptor_t* naive;

    lengths[0] = ctx->LDS_W;
    lengths[1] = ctx->BLOCK_GROUPS;
    lengths[2] = 4;

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

        /* decoded = chunk_desc.unmerge_lower(b, chunk_idx=chunk_idx) */
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
                ch_block = out_values[i]; /* "ch_block" */
            else if(out_names[i] && out_names[i][0] == 'g')
                group_in_wg = out_values[i]; /* "group_in_wg" */
            else if(out_names[i] && out_names[i][0] == 'W')
                W_lds = out_values[i]; /* "W_lds" */
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
 *  Descriptor phase (Python lines 475-519, 637-641).
 *
 *  a_desc = naive("A", [N,H,W,total_c], coord_names=(n,h,w,c))
 *           .transform(embed(("y_iter",)->"h", strides=(1,), offset=-PAD, lo=0, hi=H),
 *                      embed(("q_pos","W_lds_pos")->"w", strides=(1,1),
 *                            offset=-PAD, lo=0, hi=W))
 *  d_desc = naive("D", [N,H,W,total_k], coord_names=(n,h,w,k))
 * ===================================================================== */
void rocke_dconv16c_build_descriptors(rocke_dconv_16c_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int total_c = rocke_direct_conv_problem_total_c(&ctx->p);
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);

    /* ---- input descriptor A[N,H,W,total_c] + 2 embeds ---- */
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

        /* embed(upper=("y_iter",), into="h", strides=(1,), offset=-PAD, lo=0, hi=H) */
        {
            static const char* const h_upper[1] = {"y_iter"};
            int h_strides[1] = {1};
            xforms[0]
                = rocke_embed_bounded(b, h_upper, 1, "h", h_strides, -ctx->p.PAD, 0, ctx->p.H);
        }
        /* embed(upper=("q_pos","W_lds_pos"), into="w", strides=(1,1),
         *       offset=-PAD, lo=0, hi=W) */
        {
            static const char* const w_upper[2] = {"q_pos", "W_lds_pos"};
            int w_strides[2] = {1, 1};
            xforms[1]
                = rocke_embed_bounded(b, w_upper, 2, "w", w_strides, -ctx->p.PAD, 0, ctx->p.W);
        }
        ctx->a_desc = rocke_tensor_descriptor_transform(b, a_naive, xforms, 2);
    }

    /* ---- output descriptor D[N,H,W,total_k] naive ---- */
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
