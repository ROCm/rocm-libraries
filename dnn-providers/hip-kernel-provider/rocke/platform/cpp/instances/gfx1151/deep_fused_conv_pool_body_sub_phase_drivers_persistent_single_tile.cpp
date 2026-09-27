// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1151_deep_fused_conv_pool_Body sub-phase drivers
 *   (persistent + single-tile).c
 *
 * Chunked C99 port of the two top-level body branches of
 *   rocke/instances/gfx1151/deep_fused_conv_pool.py :: build_deep_fused_conv_pool
 * (Python lines 2780-3085). Each branch authors the full numeric body for its
 * lever family over the already-initialised builder context
 * (rocke_gfx1151_dfcp_build_ctx_t): the ctx prologue (driver) has already resolved
 * op/op0/op1, declared X/W0/Y/W1, built+bound the WarpGrid, stamped the launch
 * bound + SchedulePolicy, and allocated every LDS tile (a0/inp, w0, c0,
 * c0_packed, w1, c1). These two functions only walk the staged phases in Python
 * order, calling the peer phase functions through the internal header.
 *
 *   rocke_gfx1151_dfcp_emit_persistent_body  <- Python 2780-2876
 *       (native_int + direct_conv0 + fused_c0a1): stage W0/W1 ONCE + sync, then
 *       scf_for grid-stride over the flattened tile strip streaming X/Y per tile.
 *   rocke_gfx1151_dfcp_emit_single_tile_body <- Python 2878-3085
 *       the 4-way conv0 branch (native/fp16 x direct/im2col), conv0/conv1
 *       requant code-fn selection, early_w1 + fused_c0a1/packed_c0/repack_c0
 *       conv1 dispatch, final conv1 scatter + maxpool.
 *
 * The Python closures conv0_code_i8 / conv0_code_f16 / conv0_code_vec_i8 /
 * conv1_code_i8 / conv1_code_f16 / conv1_code_vec_i8 are realised in C as the
 * rocke_gfx1151_dfcp_code_fn_t selector enum: the scatter / fuse phases take the
 * matching selector and apply the byte-faithful emitter internally (see the
 * internal header's REQUANT CODE FUNCTIONS section), so the driver marshals no
 * function pointers. The fp16 vs native_int split of each closure is already
 * encoded in the enum value the body selects per branch.
 *
 * Byte-identical builder-call order to the Python: the b.sync() barriers, the
 * stage/gemm/scatter/maxpool sequence and the per-lever dispatch are reproduced
 * 1:1. This translation unit emits no IR of its own beyond the const_i32 /
 * scf_for / readfirstlane / div / mod scaffolding the persistent loop needs.
 */

#include "rocke/instance_gfx1151_deep_fused_conv_pool_internal.h"

#include "rocke/ir.h" /* rocke_b_* scaffolding (const_i32, scf_for, readfirstlane, div, mod) */

/* ===================================================================== *
 *  rocke_gfx1151_dfcp_emit_persistent_body  (Python 2780-2876)
 *
 *  Persistent grid-stride variant. is_valid_spec guarantees native_int +
 *  direct_conv0 + fused_c0a1, so W0/W1 are tile-invariant and there is no
 *  per-tile c0_smem: stage both weights into LDS ONCE before the loop, then
 *  grid-stride over the flattened tile strip streaming only X (footprint) and Y
 *  (output) per tile.
 *
 *  The conv0/conv1 requant closures are the *_i8 native forms; the body selects
 *  ROCKE_GFX1151_DFCP_CODE_CONV0_I8 / ROCKE_GFX1151_DFCP_CODE_CONV1_I8 for the fuse /
 *  scatter phases (Python conv0_code_i8 / conv1_code_i8 defined at 2786-2793).
 * ===================================================================== */
void rocke_gfx1151_dfcp_emit_persistent_body(rocke_gfx1151_dfcp_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec;
    const rocke_fused_conv_pool_problem_t* p;
    const rocke_conv_problem_t* c;
    int h_tiles;
    int w_tiles;
    int num_tiles;
    rocke_value_t* c_wtiles;
    rocke_for_t loop;

    if(ctx == NULL)
        return;
    b = ctx->b;
    spec = ctx->spec;
    p = ctx->p;
    c = ctx->c;

    /* Weights resident: staged once per CTA (vs once per tile), one barrier.
     *   _stage_conv0_w0_int(b, spec, W0, w0_smem, grid)
     *   conv1_int8 ? _stage_conv1_w1_i8 : _stage_conv1_w1_packed
     *   b.sync() */
    rocke_gfx1151_dfcp_stage_conv0_w0_int(ctx, ctx->W0, ctx->w0_smem);
    if(spec->conv1_int8)
        rocke_gfx1151_dfcp_stage_conv1_w1_i8(ctx, ctx->W1, ctx->w1_smem);
    else
        rocke_gfx1151_dfcp_stage_conv1_w1_packed(ctx, ctx->W1, ctx->w1_smem);
    rocke_b_sync(b);

    /* h_tiles = p.pool_ho // pool_tile_h; w_tiles = p.pool_wo // pool_tile_w;
     * num_tiles = h_tiles * w_tiles; c_wtiles = const_i32(w_tiles). */
    h_tiles = rocke_fused_conv_pool_problem_pool_ho(p) / spec->pool_tile_h;
    w_tiles = rocke_fused_conv_pool_problem_pool_wo(p) / spec->pool_tile_w;
    num_tiles = h_tiles * w_tiles;
    c_wtiles = rocke_b_const_i32(b, w_tiles);

    /* loop = b.scf_for(block_id_x(), const_i32(num_tiles),
     *                  const_i32(persistent_ctas), iv_name="tile_idx") */
    loop = rocke_b_scf_for(b,
                           rocke_b_block_id_x(b),
                           rocke_b_const_i32(b, num_tiles),
                           rocke_b_const_i32(b, spec->persistent_ctas),
                           "tile_idx");

    /* with loop as tile_idx: ... */
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* ti;

        /* Scalarize tile index + coords (uniform across wave -> SGPR):
         *   ti = readfirstlane(tile_idx)
         *   h_blk = div(ti, c_wtiles); w_blk = mod(ti, c_wtiles) */
        ti = rocke_b_readfirstlane(b, loop.iv);
        ctx->h_blk = rocke_b_div(b, ti, c_wtiles);
        ctx->w_blk = rocke_b_mod(b, ti, c_wtiles);

        /* Inter-tile guard: orders the prior tile's maxpool / conv0 GEMM
         * completions before this tile reuses a0_smem/c1_smem. */
        rocke_b_sync(b);

        /* footprint stage (persistent coords via h_blk/w_blk) -> sync */
        rocke_gfx1151_dfcp_stage_input_footprint_int(
            ctx, ctx->X, ctx->a0_smem, ctx->h_blk, ctx->w_blk);
        rocke_b_sync(b); /* footprint store -> conv0 GEMM read of a0_smem */

        /* conv0: direct native-int iu8 GEMM, reorient=True for the in-register
         * fused_c0a1 handoff (acc[k0,m]). */
        ctx->num_accs0 = 0;
        rocke_gfx1151_dfcp_wmma_gemm_conv0_direct_int(ctx,
                                                      ctx->op0,
                                                      ctx->a0_smem,
                                                      ctx->w0_smem,
                                                      ctx->policy,
                                                      /*reorient=*/true,
                                                      ctx->accs0,
                                                      &ctx->num_accs0);

        /* conv1: register handoff (no scatter/c0_smem/barrier). W1 resident +
         * never rewritten, so the pre-conv1 barrier is dropped. */
        ctx->num_a_frags = 0;
        ctx->num_accs1 = 0;
        if(spec->conv1_int8)
        {
            /* a_frags = _fuse_c0_to_conv1_a_regs(b, op0, accs0, grid,
             *                                    conv0_code_i8, int8=True) */
            rocke_gfx1151_dfcp_fuse_c0_to_conv1_a_regs(ctx,
                                                       ctx->op0,
                                                       ctx->accs0,
                                                       ctx->num_accs0,
                                                       ROCKE_GFX1151_DFCP_CODE_CONV0_I8,
                                                       /*int8=*/true,
                                                       ctx->a_frags,
                                                       &ctx->num_a_frags,
                                                       &ctx->a_frags_rows,
                                                       &ctx->a_frags_cols);
            rocke_gfx1151_dfcp_wmma_gemm_conv1_i8_from_regs(ctx,
                                                            ctx->op0,
                                                            ctx->a_frags,
                                                            ctx->a_frags_rows,
                                                            ctx->a_frags_cols,
                                                            ctx->w1_smem,
                                                            c->K,
                                                            ctx->policy,
                                                            spec->conv1_prefetch_k,
                                                            spec->conv1_sched_fuse,
                                                            ctx->accs1,
                                                            &ctx->num_accs1);
        }
        else
        {
            /* a_frags = _fuse_c0_to_conv1_a_regs(b, op0, accs0, grid,
             *                                    conv0_code_i8) */
            rocke_gfx1151_dfcp_fuse_c0_to_conv1_a_regs(ctx,
                                                       ctx->op0,
                                                       ctx->accs0,
                                                       ctx->num_accs0,
                                                       ROCKE_GFX1151_DFCP_CODE_CONV0_I8,
                                                       /*int8=*/false,
                                                       ctx->a_frags,
                                                       &ctx->num_a_frags,
                                                       &ctx->a_frags_rows,
                                                       &ctx->a_frags_cols);
            rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_from_regs(ctx,
                                                            ctx->op1,
                                                            ctx->a_frags,
                                                            ctx->a_frags_rows,
                                                            ctx->a_frags_cols,
                                                            ctx->w1_smem,
                                                            c->K,
                                                            ctx->policy,
                                                            spec->conv1_prefetch_k,
                                                            spec->conv1_sched_fuse,
                                                            ctx->accs1,
                                                            &ctx->num_accs1);
        }

        /* _scatter_codes_to_i8_lds(b, op1, accs1, c1_smem, grid, conv1_code_i8)
         * b.sync()  # conv1 scatter -> maxpool read of c1_smem
         * _emit_maxpool_finalpack_i8(b, spec, c1_smem, Y, grid, h_blk, w_blk) */
        rocke_gfx1151_dfcp_scatter_codes_to_i8_lds(ctx,
                                                   ctx->op1,
                                                   ctx->accs1,
                                                   ctx->num_accs1,
                                                   ctx->c1_smem,
                                                   ROCKE_GFX1151_DFCP_CODE_CONV1_I8);
        rocke_b_sync(b);
        rocke_gfx1151_dfcp_emit_maxpool_finalpack_i8(
            ctx, ctx->c1_smem, ctx->Y, ctx->h_blk, ctx->w_blk);
    }
    rocke_b_region_leave(b);

    /* Clear loop-carried coords; subsequent non-persistent phases (none here)
     * must fall back to block ids. */
    ctx->h_blk = NULL;
    ctx->w_blk = NULL;
}

/* ===================================================================== *
 *  rocke_gfx1151_dfcp_emit_single_tile_body  (Python 2878-3085)
 *
 *  Single-tile pipeline: conv0 (direct/im2col x native/fp16) -> requant scatter
 *  / register handoff -> conv1 (iu4/iu8/fp16) -> requant scatter -> maxpool
 *  (native finalpack / fp16 finalquant). All staged over the per-CTA block ids
 *  (h_blk/w_blk left NULL so the _int phases use the block-id fallback).
 * ===================================================================== */
void rocke_gfx1151_dfcp_emit_single_tile_body(rocke_gfx1151_dfcp_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b;
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec;
    const rocke_conv_problem_t* c;
    /* The conv0 / conv1 requant code-fn selectors chosen per lever, mirroring
     * the Python closures defined at 2919-2940 (conv0) and 3051-3066 (conv1). */
    int conv0_code; /* scalar conv0 code-fn (i8 / f16)            */
    int conv0_code_vec; /* vector conv0 code-fn (specialized_rne)     */
    int conv1_code; /* scalar conv1 code-fn (i8 / f16)            */
    int conv1_code_vec; /* vector conv1 code-fn (specialized_rne)     */

    if(ctx == NULL)
        return;
    b = ctx->b;
    spec = ctx->spec;
    c = ctx->c;

    /* Single-tile path: no persistent loop-carried coords. */
    ctx->h_blk = NULL;
    ctx->w_blk = NULL;

    /* Per-lever requant code-fn selection. native_int picks the *_i8 forms;
     * the fp16 path the *_f16 forms (which call the *_i8 closure then truncate
     * to f16). conv0_code_vec_i8 / conv1_code_vec_i8 feed the specialized_rne
     * whole-accumulator scatter path. */
    if(spec->native_int)
    {
        conv0_code = ROCKE_GFX1151_DFCP_CODE_CONV0_I8;
        conv1_code = ROCKE_GFX1151_DFCP_CODE_CONV1_I8;
    }
    else
    {
        conv0_code = ROCKE_GFX1151_DFCP_CODE_CONV0_F16;
        conv1_code = ROCKE_GFX1151_DFCP_CODE_CONV1_F16;
    }
    conv0_code_vec = ROCKE_GFX1151_DFCP_CODE_CONV0_VEC_I8;
    conv1_code_vec = ROCKE_GFX1151_DFCP_CODE_CONV1_VEC_I8;

    /* ---- conv0: int8 -> WMMA -> Quant(i32->i8)->ReLU->Quant(i8->i4) ----
     * Four-way branch: native/fp16 x direct/im2col (Python 2881-2913). */
    ctx->num_accs0 = 0;
    if(spec->native_int && spec->direct_conv0)
    {
        /* _stage_input_footprint_int(b, spec, X, a0_smem, grid)
         * _stage_conv0_w0_int(b, spec, W0, w0_smem, grid)
         * b.sync()
         * accs0 = _wmma_gemm_conv0_direct_int(..., reorient=spec.fused_c0a1) */
        rocke_gfx1151_dfcp_stage_input_footprint_int(ctx,
                                                     ctx->X,
                                                     ctx->a0_smem,
                                                     /*h_blk=*/NULL,
                                                     /*w_blk=*/NULL);
        rocke_gfx1151_dfcp_stage_conv0_w0_int(ctx, ctx->W0, ctx->w0_smem);
        rocke_b_sync(b);
        rocke_gfx1151_dfcp_wmma_gemm_conv0_direct_int(ctx,
                                                      ctx->op0,
                                                      ctx->a0_smem,
                                                      ctx->w0_smem,
                                                      ctx->policy,
                                                      /*reorient=*/spec->fused_c0a1,
                                                      ctx->accs0,
                                                      &ctx->num_accs0);
    }
    else if(spec->native_int)
    {
        /* _stage_conv0_a_int / _stage_conv0_w0_int / b.sync()
         * accs0 = _wmma_gemm_from_lds_int(b, op0, a0_smem, w0_smem, grid, kpad) */
        rocke_gfx1151_dfcp_stage_conv0_a_int(ctx, ctx->X, ctx->a0_smem);
        rocke_gfx1151_dfcp_stage_conv0_w0_int(ctx, ctx->W0, ctx->w0_smem);
        rocke_b_sync(b);
        rocke_gfx1151_dfcp_wmma_gemm_from_lds_int(ctx,
                                                  ctx->op0,
                                                  ctx->a0_smem,
                                                  ctx->w0_smem,
                                                  ctx->kpad,
                                                  ctx->policy,
                                                  ctx->accs0,
                                                  &ctx->num_accs0);
    }
    else if(spec->direct_conv0)
    {
        /* _stage_input_footprint / _stage_conv0_w0 / b.sync()
         * accs0 = _wmma_gemm_conv0_direct(b, spec, op, a0_smem, w0_smem, grid) */
        rocke_gfx1151_dfcp_stage_input_footprint(ctx, ctx->X, ctx->a0_smem);
        rocke_gfx1151_dfcp_stage_conv0_w0(ctx, ctx->W0, ctx->w0_smem);
        rocke_b_sync(b);
        rocke_gfx1151_dfcp_wmma_gemm_conv0_direct(
            ctx, ctx->op, ctx->a0_smem, ctx->w0_smem, ctx->policy, ctx->accs0, &ctx->num_accs0);
    }
    else
    {
        /* _stage_conv0_a / _stage_conv0_w0 / b.sync()
         * accs0 = _wmma_gemm_from_lds(b, op, a0_smem, w0_smem, grid, kpad) */
        rocke_gfx1151_dfcp_stage_conv0_a(ctx, ctx->X, ctx->a0_smem);
        rocke_gfx1151_dfcp_stage_conv0_w0(ctx, ctx->W0, ctx->w0_smem);
        rocke_b_sync(b);
        rocke_gfx1151_dfcp_wmma_gemm_from_lds(ctx,
                                              ctx->op,
                                              ctx->a0_smem,
                                              ctx->w0_smem,
                                              ctx->kpad,
                                              ctx->policy,
                                              ctx->accs0,
                                              &ctx->num_accs0);
    }

    /* Requant closure constants hoisted to the body level (Python 2915-2917):
     *   c_m0 = const_f32(m0); c_m0b = const_f32(m0b); zero_f = const_f32(0.0)
     * Created here -- after the conv0 GEMM, before the conv1 staging/scatter --
     * so the f32 const SSA ids match the Python closure-capture order. The
     * code-fn emitters read ctx->c_m0/c_m0b/zero_f instead of re-creating them. */
    ctx->c_m0 = rocke_b_const_f32(b, (double)spec->m0);
    ctx->c_m0b = rocke_b_const_f32(b, (double)spec->m0b);
    ctx->zero_f = rocke_b_const_f32(b, 0.0);

    /* ---- conv1: 1x1 int4 -> int32/fp32 -> Quant(i32->i4)->ReLU ----
     * (Python 2942-3047). accs0 are in registers; the conv1 operand tiles are
     * distinct from conv0's, so no barrier is needed before producing them. */
    ctx->num_a_frags = 0;
    ctx->num_accs1 = 0;
    if(spec->native_int && spec->fused_c0a1)
    {
        /* In-register handoff: no scatter, no c0_smem, no handoff barrier. */
        if(spec->conv1_int8)
        {
            /* int8 conv1: byte-per-code W1 + unsqueezed <4 x i32> A-frags + iu8. */
            rocke_gfx1151_dfcp_stage_conv1_w1_i8(ctx, ctx->W1, ctx->w1_smem);
            rocke_gfx1151_dfcp_fuse_c0_to_conv1_a_regs(ctx,
                                                       ctx->op0,
                                                       ctx->accs0,
                                                       ctx->num_accs0,
                                                       conv0_code,
                                                       /*int8=*/true,
                                                       ctx->a_frags,
                                                       &ctx->num_a_frags,
                                                       &ctx->a_frags_rows,
                                                       &ctx->a_frags_cols);
            rocke_b_sync(b);
            rocke_gfx1151_dfcp_wmma_gemm_conv1_i8_from_regs(ctx,
                                                            ctx->op0,
                                                            ctx->a_frags,
                                                            ctx->a_frags_rows,
                                                            ctx->a_frags_cols,
                                                            ctx->w1_smem,
                                                            c->K,
                                                            ctx->policy,
                                                            spec->conv1_prefetch_k,
                                                            spec->conv1_sched_fuse,
                                                            ctx->accs1,
                                                            &ctx->num_accs1);
        }
        else
        {
            rocke_gfx1151_dfcp_stage_conv1_w1_packed(ctx, ctx->W1, ctx->w1_smem);
            rocke_gfx1151_dfcp_fuse_c0_to_conv1_a_regs(ctx,
                                                       ctx->op0,
                                                       ctx->accs0,
                                                       ctx->num_accs0,
                                                       conv0_code,
                                                       /*int8=*/false,
                                                       ctx->a_frags,
                                                       &ctx->num_a_frags,
                                                       &ctx->a_frags_rows,
                                                       &ctx->a_frags_cols);
            rocke_b_sync(b);
            rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_from_regs(ctx,
                                                            ctx->op1,
                                                            ctx->a_frags,
                                                            ctx->a_frags_rows,
                                                            ctx->a_frags_cols,
                                                            ctx->w1_smem,
                                                            c->K,
                                                            ctx->policy,
                                                            spec->conv1_prefetch_k,
                                                            spec->conv1_sched_fuse,
                                                            ctx->accs1,
                                                            &ctx->num_accs1);
        }
    }
    else if(spec->native_int)
    {
        /* early_w1 issues the W1 HBM loads before the conv0 epilogue scatter so
         * their latency overlaps; a single barrier then gates conv1 on both. */
        if(spec->early_w1)
        {
            rocke_gfx1151_dfcp_stage_conv1_w1_packed(ctx, ctx->W1, ctx->w1_smem);
            if(spec->packed_c0_handoff)
                rocke_gfx1151_dfcp_scatter_packed_i4_codes_to_lds(
                    ctx, ctx->op0, ctx->accs0, ctx->num_accs0, ctx->c0_smem, conv0_code);
            else if(spec->specialized_rne)
                rocke_gfx1151_dfcp_scatter_vec_codes_to_i8_lds(
                    ctx, ctx->op0, ctx->accs0, ctx->num_accs0, ctx->c0_smem, conv0_code_vec);
            else
                rocke_gfx1151_dfcp_scatter_codes_to_i8_lds(
                    ctx, ctx->op0, ctx->accs0, ctx->num_accs0, ctx->c0_smem, conv0_code);
            rocke_b_sync(b);
        }
        else
        {
            rocke_b_sync(b);
            if(spec->packed_c0_handoff)
                rocke_gfx1151_dfcp_scatter_packed_i4_codes_to_lds(
                    ctx, ctx->op0, ctx->accs0, ctx->num_accs0, ctx->c0_smem, conv0_code);
            else if(spec->specialized_rne)
                rocke_gfx1151_dfcp_scatter_vec_codes_to_i8_lds(
                    ctx, ctx->op0, ctx->accs0, ctx->num_accs0, ctx->c0_smem, conv0_code_vec);
            else
                rocke_gfx1151_dfcp_scatter_codes_to_i8_lds(
                    ctx, ctx->op0, ctx->accs0, ctx->num_accs0, ctx->c0_smem, conv0_code);
            rocke_gfx1151_dfcp_stage_conv1_w1_packed(ctx, ctx->W1, ctx->w1_smem);
            rocke_b_sync(b);
        }

        if(spec->repack_c0)
        {
            rocke_gfx1151_dfcp_repack_c0_lds_to_packed(ctx, ctx->c0_smem, ctx->c0_packed_smem);
            rocke_b_sync(b);
            rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_packed_from_lds(ctx,
                                                                  ctx->op1,
                                                                  ctx->c0_packed_smem,
                                                                  ctx->w1_smem,
                                                                  c->K,
                                                                  ctx->policy,
                                                                  ctx->accs1,
                                                                  &ctx->num_accs1);
        }
        else if(spec->packed_c0_handoff)
        {
            rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_packed_from_lds(ctx,
                                                                  ctx->op1,
                                                                  ctx->c0_smem,
                                                                  ctx->w1_smem,
                                                                  c->K,
                                                                  ctx->policy,
                                                                  ctx->accs1,
                                                                  &ctx->num_accs1);
        }
        else
        {
            rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_from_lds(ctx,
                                                           ctx->op1,
                                                           ctx->c0_smem,
                                                           ctx->w1_smem,
                                                           c->K,
                                                           ctx->policy,
                                                           spec->conv1_prefetch_k,
                                                           spec->conv1_sched_fuse,
                                                           ctx->accs1,
                                                           &ctx->num_accs1);
        }
    }
    else
    {
        /* fp16 path. */
        if(spec->early_w1)
        {
            rocke_gfx1151_dfcp_stage_conv1_w1(ctx, ctx->W1, ctx->w1_smem);
            rocke_gfx1151_dfcp_scatter_codes_to_lds(
                ctx, ctx->op0, ctx->accs0, ctx->num_accs0, ctx->c0_smem, conv0_code);
            rocke_b_sync(b);
        }
        else
        {
            rocke_b_sync(b);
            rocke_gfx1151_dfcp_scatter_codes_to_lds(
                ctx, ctx->op0, ctx->accs0, ctx->num_accs0, ctx->c0_smem, conv0_code);
            rocke_gfx1151_dfcp_stage_conv1_w1(ctx, ctx->W1, ctx->w1_smem);
            rocke_b_sync(b);
        }
        rocke_gfx1151_dfcp_wmma_gemm_from_lds(ctx,
                                              ctx->op,
                                              ctx->c0_smem,
                                              ctx->w1_smem,
                                              c->K,
                                              ctx->policy,
                                              ctx->accs1,
                                              &ctx->num_accs1);
    }

    /* c_m1 = const_f32(m1) (Python 3049): hoisted after the conv1 GEMM, before
     * the conv1 requant scatter, so its f32 const id matches the closure. */
    ctx->c_m1 = rocke_b_const_f32(b, (double)spec->m1);

    /* ---- conv1 requant scatter -> c1_smem (Python 3068-3077) ---- */
    if(spec->native_int)
    {
        if(spec->specialized_rne)
            rocke_gfx1151_dfcp_scatter_vec_codes_to_i8_lds(
                ctx, ctx->op1, ctx->accs1, ctx->num_accs1, ctx->c1_smem, conv1_code_vec);
        else
            rocke_gfx1151_dfcp_scatter_codes_to_i8_lds(
                ctx, ctx->op1, ctx->accs1, ctx->num_accs1, ctx->c1_smem, conv1_code);
    }
    else
    {
        rocke_gfx1151_dfcp_scatter_codes_to_lds(
            ctx, ctx->op, ctx->accs1, ctx->num_accs1, ctx->c1_smem, conv1_code);
    }
    rocke_b_sync(b);

    /* ---- maxpool 2x2/s2 -> Quant(i4->i4) -> packed int4 output
     *      (Python 3079-3083) ---- */
    if(spec->native_int)
        rocke_gfx1151_dfcp_emit_maxpool_finalpack_i8(ctx,
                                                     ctx->c1_smem,
                                                     ctx->Y,
                                                     /*h_blk=*/NULL,
                                                     /*w_blk=*/NULL);
    else
        rocke_gfx1151_dfcp_emit_maxpool_finalquant(ctx, ctx->c1_smem, ctx->Y);
}
