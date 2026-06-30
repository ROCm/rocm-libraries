// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_implicit_gemm_conv_epilogue.c -- C99 port of the accumulator +
 * store epilogues of build_implicit_gemm_conv
 * (rocke/instances/common/conv_implicit_gemm.py).
 *
 * Scope (byte-identical builder-call sequence to the Python source spans):
 *   rocke_conv_apply_accumulator_epilogue   <- _apply_accumulator_epilogue  (647-689)
 *   rocke_conv_emit_epilogue                <- epilogue dispatcher          (1349-1377)
 *   rocke_conv_emit_direct_epilogue         <- _emit_direct_epilogue        (1386-1414)
 *   rocke_conv_emit_direct_epilogue_wmma    <- _emit_direct_epilogue_wmma   (1417-1478)
 *   rocke_conv_emit_cshuffle_epilogue       <- _emit_cshuffle_epilogue      (1481-1523)
 *
 * Peers (descriptor builders, the epilogue helpers, the MMA op layout maps) are
 * reached through the public/internal headers; this TU touches only ctx + the
 * builder it carries and the value-type helpers it includes.
 */
#include <string.h>

#include "rocke/instance_conv_implicit_gemm_internal.h"

/* ===================================================================== *
 * rocke_conv_apply_accumulator_epilogue   (Python lines 647-689)
 *
 * Apply a static fp32 epilogue to each accumulator fragment. The transform is
 * scalar per accumulator lane, then packed back into the original vector width
 * so the existing direct/cshuffle epilogues can consume the result unchanged.
 * Identity copies through. Writes `num_accs` results into out_accs.
 * ===================================================================== */
void rocke_conv_apply_accumulator_epilogue(rocke_ir_builder_t* b,
                                           const rocke_conv_acc_epilogue_t* epilogue,
                                           rocke_value_t* const* accs,
                                           int num_accs,
                                           rocke_value_t** out_accs)
{
    int a, i;
    rocke_value_t* c_zero;
    rocke_value_t* c_bias;
    rocke_value_t* c_scale;
    rocke_value_t* c_clamp_min;
    rocke_value_t* c_clamp_max;

    /* if epilogue.is_identity(): return list(accs) */
    if(rocke_conv_acc_epilogue_is_identity(epilogue))
    {
        for(a = 0; a < num_accs; ++a)
            out_accs[a] = accs[a];
        return;
    }

    /* c_zero = b.const_f32(0.0) */
    c_zero = rocke_b_const_f32(b, 0.0);
    /* c_bias = b.const_f32(epilogue.bias) if epilogue.bias != 0.0 else None */
    c_bias = (epilogue->bias != 0.0) ? rocke_b_const_f32(b, epilogue->bias) : NULL;
    /* c_scale = b.const_f32(epilogue.scale) if epilogue.scale != 1.0 else None */
    c_scale = (epilogue->scale != 1.0) ? rocke_b_const_f32(b, epilogue->scale) : NULL;
    /* c_clamp_min = b.const_f32(epilogue.clamp_min) if clamp_min is not None else None */
    c_clamp_min = epilogue->has_clamp_min ? rocke_b_const_f32(b, epilogue->clamp_min) : NULL;
    /* c_clamp_max = b.const_f32(epilogue.clamp_max) if clamp_max is not None else None */
    c_clamp_max = epilogue->has_clamp_max ? rocke_b_const_f32(b, epilogue->clamp_max) : NULL;

    for(a = 0; a < num_accs; ++a)
    {
        rocke_value_t* acc = accs[a];
        int count = acc->type->count;
        /* count is the per-lane accumulator fragment width (c_frag_len: 4/8/16);
         * 64 is generous headroom. */
        rocke_value_t* elems[64];
        for(i = 0; i < count; ++i)
        {
            /* v = b.vec_extract(acc, i) */
            rocke_value_t* v = rocke_b_vec_extract(b, acc, i);
            if(c_bias != NULL)
                v = rocke_b_fadd(b, v, c_bias);
            if(c_scale != NULL)
                v = rocke_b_fmul(b, v, c_scale);
            if(epilogue->relu)
                v = rocke_b_fmax(b, v, c_zero);
            if(c_clamp_min != NULL)
                v = rocke_b_fmax(b, v, c_clamp_min);
            if(c_clamp_max != NULL)
                v = rocke_b_fmin(b, v, c_clamp_max);
            elems[i] = v;
        }
        /* out.append(b.vec_pack(elems, elems[0].type)) */
        out_accs[a] = rocke_b_vec_pack(b, elems, count, elems[0]->type);
    }
}

/* ===================================================================== *
 * D-descriptor address closure shared by the direct + cshuffle stores.
 *
 * Python:
 *   def d_addr(b_, m_val, n_val):
 *       return D_desc.offset(b_, m=m_val, k_out=n_val)
 *
 * The rocke_epilogue_addr_fn contract returns off_elements directly and writes
 * the optional i1 validity through *out_valid. `user` carries the bound
 * D descriptor. */
static rocke_value_t* rocke_conv_d_addr(rocke_ir_builder_t* b,
                                        rocke_value_t* m_global,
                                        rocke_value_t* n_global,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    const rocke_tensor_descriptor_t* D_desc = (const rocke_tensor_descriptor_t*)user;
    const char* names[2];
    rocke_value_t* values[2];
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;

    names[0] = "m";
    names[1] = "k_out";
    values[0] = m_global;
    values[1] = n_global;

    rocke_transforms_descriptor_offset(b, D_desc, names, values, 2, &off, &valid);
    if(out_valid != NULL)
        *out_valid = valid;
    return off;
}

/* ===================================================================== *
 * rocke_conv_emit_direct_epilogue   (Python lines 1386-1414)
 *
 * Per-lane scalar-fp16 store driven by the D descriptor DAG. Delegates to
 * DirectEpilogue.store; the conv-specific bit is the addr_fn that maps
 * (m, k_out) -> NHWK linear element offset.
 * ===================================================================== */
void rocke_conv_emit_direct_epilogue(rocke_ir_builder_t* b,
                                     const rocke_implicit_gemm_conv_spec_t* spec,
                                     rocke_value_t* const* accs,
                                     int num_accs,
                                     const rocke_warp_grid_t* grid,
                                     rocke_value_t* d_rsrc)
{
    const rocke_conv_problem_t* p = &spec->problem;
    /* D_desc = make_d_descriptor(p) */
    rocke_tensor_descriptor_t* D_desc = rocke_conv_make_d_descriptor(b, p);
    rocke_direct_epilogue_t epi;

    /* DirectEpilogue(atom=spec.atom, grid=grid).store(
     *     b, accs=accs, addr_fn=d_addr, d_rsrc=d_rsrc,
     *     bounds=(b.const_i32(p.M), b.const_i32(p.N_gemm))) */
    epi.atom = rocke_mfma_atom("f16", spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    epi.grid = *grid;

    {
        /* hoist bounds in Python's left-to-right order: M first, then N_gemm */
        rocke_value_t* bound_m = rocke_b_const_i32(b, rocke_conv_problem_m(p));
        rocke_value_t* bound_n = rocke_b_const_i32(b, rocke_conv_problem_n_gemm(p));
        rocke_direct_epilogue_store(b,
                                    &epi,
                                    accs,
                                    num_accs,
                                    rocke_conv_d_addr,
                                    (void*)D_desc,
                                    d_rsrc,
                                    bound_m,
                                    bound_n,
                                    false);
    }
}

/* ===================================================================== *
 * rocke_conv_emit_direct_epilogue_wmma   (Python lines 1417-1478)
 *
 * Per-lane fp16 store for the WMMA (gfx1151) accumulator layout. The (row, col)
 * of every per-lane slot comes from the op's accumulator layout map
 * (op.c_layout()) rather than the MFMA-specific MfmaAtom.lane_to_output. Each
 * slot is one f16 store routed through the same D descriptor + OOB-safe
 * buffer-store idiom as the MFMA direct epilogue.
 * ===================================================================== */
void rocke_conv_emit_direct_epilogue_wmma(rocke_ir_builder_t* b,
                                          const rocke_implicit_gemm_conv_spec_t* spec,
                                          const rocke_mmaop_t* op,
                                          rocke_value_t* const* accs,
                                          int num_accs,
                                          rocke_value_t* warp_m_idx,
                                          rocke_value_t* warp_n_idx,
                                          rocke_value_t* lane,
                                          rocke_value_t* block_m_off,
                                          rocke_value_t* block_n_off,
                                          rocke_value_t* d_rsrc,
                                          rocke_value_t* c0)
{
    const rocke_conv_problem_t* p = &spec->problem;
    int mfmas_m = rocke_implicit_gemm_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_implicit_gemm_conv_spec_mfmas_per_warp_n(spec);

    /* warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m)) */
    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    /* warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n)) */
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));

    /* c_M = b.const_i32(p.M); c_N = b.const_i32(p.N_gemm) */
    rocke_value_t* c_M = rocke_b_const_i32(b, rocke_conv_problem_m(p));
    rocke_value_t* c_N = rocke_b_const_i32(b, rocke_conv_problem_n_gemm(p));
    /* D_desc = make_d_descriptor(p) */
    rocke_tensor_descriptor_t* D_desc = rocke_conv_make_d_descriptor(b, p);
    /* c_map = op.c_layout() */
    const rocke_arch_layout_map_t* c_map = rocke_mmaop_c_layout(op, b);

    int flat = 0;
    int mi, ni, i;
    (void)num_accs;

    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = accs[flat];
            rocke_value_t* atom_m_off;
            rocke_value_t* atom_n_off;
            /* C evaluates nested call args in unspecified (typically
             * right-to-left) order, which would create the const before the
             * inner add and swap their SSA ids vs Python (which evaluates the
             * inner b.add first, then the b.const_i32). Bind each subexpression
             * to a temp in Python's left-to-right order. */
            rocke_value_t* m_inner;
            rocke_value_t* m_const;
            rocke_value_t* n_inner;
            rocke_value_t* n_const;
            ++flat;

            /* atom_m_off = b.add(b.add(block_m_off, warp_m_off),
             *                    b.const_i32(mi * spec.warp_tile_m)) */
            m_inner = rocke_b_add(b, block_m_off, warp_m_off);
            m_const = rocke_b_const_i32(b, mi * spec->warp_tile_m);
            atom_m_off = rocke_b_add(b, m_inner, m_const);
            /* atom_n_off = b.add(b.add(block_n_off, warp_n_off),
             *                    b.const_i32(ni * spec.warp_tile_n)) */
            n_inner = rocke_b_add(b, block_n_off, warp_n_off);
            n_const = rocke_b_const_i32(b, ni * spec->warp_tile_n);
            atom_n_off = rocke_b_add(b, n_inner, n_const);

            for(i = 0; i < op->c_frag_len; ++i)
            {
                rocke_value_t* row_off = NULL;
                rocke_value_t* col_off = NULL;
                rocke_value_t* m_val;
                rocke_value_t* n_val;
                rocke_value_t* m_ok;
                rocke_value_t* n_ok;
                rocke_value_t* ok;
                rocke_value_t* v_f32;
                rocke_value_t* v_f16;
                rocke_value_t* d_off_elems = NULL;
                rocke_value_t* d_off_bytes;
                rocke_value_t* safe_off;

                /* row_off, col_off = c_map.coord(b, lane, i) */
                rocke_arch_layout_map_coord(c_map, b, lane, i, &row_off, &col_off);
                /* m_val = b.add(atom_m_off, row_off) */
                m_val = rocke_b_add(b, atom_m_off, row_off);
                /* n_val = b.add(atom_n_off, col_off) */
                n_val = rocke_b_add(b, atom_n_off, col_off);
                /* m_ok = b.cmp_lt(m_val, c_M); n_ok = b.cmp_lt(n_val, c_N) */
                m_ok = rocke_b_cmp_lt(b, m_val, c_M);
                n_ok = rocke_b_cmp_lt(b, n_val, c_N);
                /* ok = b.land(m_ok, n_ok) */
                ok = rocke_b_land(b, m_ok, n_ok);

                /* v_f32 = b.vec_extract(acc, i) */
                v_f32 = rocke_b_vec_extract(b, acc, i);
                /* v_f16 = b.trunc_f32_to_f16(v_f32) */
                v_f16 = rocke_b_trunc_f32_to_f16(b, v_f32);

                /* d_off_elems, _ = D_desc.offset(b, m=m_val, k_out=n_val) */
                {
                    const char* names[2];
                    rocke_value_t* values[2];
                    rocke_value_t* valid = NULL;
                    names[0] = "m";
                    names[1] = "k_out";
                    values[0] = m_val;
                    values[1] = n_val;
                    rocke_transforms_descriptor_offset(
                        b, D_desc, names, values, 2, &d_off_elems, &valid);
                }
                /* d_off_bytes = b.mul(d_off_elems, b.const_i32(2)) */
                d_off_bytes = rocke_b_mul(b, d_off_elems, rocke_b_const_i32(b, 2));
                /* safe_off = b.select(ok, d_off_bytes, b.const_i32((1<<31)-1)) */
                safe_off = rocke_b_select(
                    b, ok, d_off_bytes, rocke_b_const_i32(b, (int64_t)((1u << 31) - 1u)));
                /* b.buffer_store_f16(d_rsrc, safe_off, c0, v_f16) */
                rocke_b_buffer_store_f16(b, d_rsrc, safe_off, c0, v_f16);
            }
        }
    }
}

/* ===================================================================== *
 * rocke_conv_emit_cshuffle_epilogue   (Python lines 1481-1523)
 *
 * LDS-staged cshuffle store via CShuffleEpilogue.from_grid; the conv-specific
 * bit is the same D-descriptor addr_fn used by the direct path.
 * ===================================================================== */
void rocke_conv_emit_cshuffle_epilogue(rocke_ir_builder_t* b,
                                       const rocke_implicit_gemm_conv_spec_t* spec,
                                       rocke_value_t* const* accs,
                                       int num_accs,
                                       const rocke_warp_grid_t* grid,
                                       rocke_value_t* d_rsrc)
{
    const rocke_conv_problem_t* p = &spec->problem;
    /* D_desc = make_d_descriptor(p) */
    rocke_tensor_descriptor_t* D_desc = rocke_conv_make_d_descriptor(b, p);
    const rocke_mfma_atom_t* atom
        = rocke_mfma_atom("f16", spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    /* CShuffleEpilogue.from_grid(atom=spec.atom, grid=grid, **kwargs)
     * #8624: max_store_vec = spec.vector_size_c if not None else default 8. */
    int max_store_vec = spec->has_vector_size_c ? spec->vector_size_c : 8;
    rocke_cshuffle_epilogue_t epi = rocke_cshuffle_epilogue_from_grid(atom, grid, max_store_vec);

    /* .store(b, accs=accs, addr_fn=d_addr, d_rsrc=d_rsrc,
     *        bounds=(b.const_i32(p.M), b.const_i32(p.N_gemm))) */
    {
        /* hoist bounds in Python's left-to-right order: M first, then N_gemm */
        rocke_value_t* bound_m = rocke_b_const_i32(b, rocke_conv_problem_m(p));
        rocke_value_t* bound_n = rocke_b_const_i32(b, rocke_conv_problem_n_gemm(p));
        rocke_cshuffle_epilogue_store(
            b, &epi, accs, num_accs, rocke_conv_d_addr, (void*)D_desc, d_rsrc, bound_m, bound_n);
    }
}

/* ===================================================================== *
 * rocke_conv_emit_epilogue   (Python lines 1349-1377)
 *
 * The epilogue phase: apply the accumulator epilogue, then dispatch
 * epilogue_override / cshuffle / wmma-direct / mfma-direct exactly as the
 * Python if/elif chain. Reads ctx->final_accs; builds ctx->D_desc as the
 * per-epilogue helpers need it.
 * ===================================================================== */
void rocke_conv_emit_epilogue(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    int num_accs = ctx->num_final_accs;
    int i;

    /* final_accs = _apply_accumulator_epilogue(b, spec.acc_epilogue, final_accs) */
    rocke_value_t* final_accs[ROCKE_CONV_MAX_ACCS];
    rocke_conv_apply_accumulator_epilogue(
        b, &spec->acc_epilogue, ctx->final_accs, num_accs, final_accs);
    for(i = 0; i < num_accs; ++i)
        ctx->final_accs[i] = final_accs[i];

    /* if epilogue_override is not None: */
    if(ctx->ov != NULL && ctx->ov->epilogue_override != NULL)
    {
        /* epilogue_override(b, spec, final_accs, grid, d_rsrc, extra_context) */
        ctx->ov->epilogue_override(b,
                                   spec,
                                   final_accs,
                                   num_accs,
                                   &ctx->grid,
                                   ctx->d_rsrc,
                                   ctx->extra_context,
                                   ctx->ov->user);
    }
    /* elif spec.epilogue == "cshuffle": */
    else if(spec->epilogue != NULL && strcmp(spec->epilogue, "cshuffle") == 0)
    {
        rocke_conv_emit_cshuffle_epilogue(b, spec, final_accs, num_accs, &ctx->grid, ctx->d_rsrc);
    }
    /* elif op.family == "wmma": */
    else if(ctx->op != NULL && ctx->op->family != NULL && strcmp(ctx->op->family, "wmma") == 0)
    {
        rocke_conv_emit_direct_epilogue_wmma(b,
                                             spec,
                                             ctx->op,
                                             final_accs,
                                             num_accs,
                                             ctx->warp_m_idx,
                                             ctx->warp_n_idx,
                                             ctx->lane,
                                             ctx->block_m_off_v,
                                             ctx->block_n_off_v,
                                             ctx->d_rsrc,
                                             ctx->c0);
    }
    /* else: */
    else
    {
        rocke_conv_emit_direct_epilogue(b, spec, final_accs, num_accs, &ctx->grid, ctx->d_rsrc);
    }
}
