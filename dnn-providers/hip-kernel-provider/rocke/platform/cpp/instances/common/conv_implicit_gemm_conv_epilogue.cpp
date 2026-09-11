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
 * Python (grouped conv path, PR #10064):
 *   k_out_group_base = b.mul(b.block_id_z(), b.const_i32(p.kpg)) if p.groups>1 else None
 *   def d_addr(b_, m_val, n_val):
 *       k_out = n_val if k_out_group_base is None else b_.add(k_out_group_base, n_val)
 *       return D_desc.offset(b_, m=m_val, k_out=k_out)
 *
 * `user` carries a rocke_conv_d_addr_ctx_t holding the D descriptor and the
 * pre-computed k_out_group_base (NULL for groups==1). */
typedef struct rocke_conv_d_addr_ctx
{
    const rocke_tensor_descriptor_t* D_desc;
    rocke_value_t* k_out_group_base; /* NULL => groups==1 */
} rocke_conv_d_addr_ctx_t;

static rocke_value_t* rocke_conv_d_addr(rocke_ir_builder_t* b,
                                        rocke_value_t* m_global,
                                        rocke_value_t* n_global,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    const rocke_conv_d_addr_ctx_t* dctx = (const rocke_conv_d_addr_ctx_t*)user;
    const char* names[2];
    rocke_value_t* values[2];
    rocke_value_t* k_out;
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;

    /* k_out = k_out_group_base + n_val (grouped) or n_val (ungrouped) */
    k_out = (dctx->k_out_group_base != NULL) ? rocke_b_add(b, dctx->k_out_group_base, n_global)
                                             : n_global;

    names[0] = "m";
    names[1] = "k_out";
    values[0] = m_global;
    values[1] = k_out;

    rocke_transforms_descriptor_offset(b, dctx->D_desc, names, values, 2, &off, &valid);
    if(out_valid != NULL)
        *out_valid = valid;
    return off;
}

/* Build k_out_group_base = b.mul(b.block_id_z(), b.const_i32(kpg)) when groups>1.
 * Returns NULL for groups==1 (byte-identical ungrouped path).
 * Python evaluates b.block_id_z() (left arg) BEFORE b.const_i32(kpg) (right arg).
 * Bind each subexpression to a temp to force left-to-right SSA emission. */
static rocke_value_t* rocke_conv_make_k_out_group_base(rocke_ir_builder_t* b,
                                                       const rocke_conv_problem_t* p)
{
    if(p->groups <= 1)
        return NULL;
    rocke_value_t* bid_z = rocke_b_block_id_z(b);
    rocke_value_t* c_kpg = rocke_b_const_i32(b, rocke_conv_problem_kpg(p));
    return rocke_b_mul(b, bid_z, c_kpg);
}

/* Pointwise D-address closure: flat offset = m * kpg + n, always valid.
 * Python: def d_addr(b_, m_val, n_val): return b_.add(b_.mul(m_val, _c_K_ir), n_val), 1 */
static rocke_value_t* rocke_conv_d_addr_pointwise(rocke_ir_builder_t* b,
                                                  rocke_value_t* m_global,
                                                  rocke_value_t* n_global,
                                                  rocke_value_t** out_valid,
                                                  void* user)
{
    rocke_value_t* c_K = (rocke_value_t*)user;
    rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, m_global, c_K), n_global);
    if(out_valid != NULL)
        *out_valid = rocke_b_const_i32(b, 1);
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
                                     rocke_value_t* d_rsrc,
                                     rocke_value_t* ir_c_K_pw)
{
    const rocke_conv_problem_t* p = &spec->problem;
    rocke_direct_epilogue_t epi;

    epi.atom = rocke_mfma_atom("f16", spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    epi.grid = *grid;
    epi.out_dtype = spec->dtype_d;

    if(rocke_conv_problem_is_pointwise(p))
    {
        /* Pointwise fast path: flat offset = m * kpg + n, always valid.
         * Python _emit_direct_epilogue emits _c_K_ir FIRST, then bound_m/bound_n:
         *   _c_K_ir  = b.const_i32(p.kpg)       <- first
         *   bound_m  = b.const_i32(p.M)          <- second (inside bounds= arg)
         *   bound_n  = b.const_i32(p.N_gemm)     <- third
         * Match this order exactly. */
        rocke_value_t* c_K = rocke_b_const_i32(b, rocke_conv_problem_kpg(p));
        rocke_value_t* bound_m = rocke_b_const_i32(b, rocke_conv_problem_m(p));
        rocke_value_t* bound_n = rocke_b_const_i32(b, rocke_conv_problem_n_gemm(p));
        (void)ir_c_K_pw;
        rocke_direct_epilogue_store(b,
                                    &epi,
                                    accs,
                                    num_accs,
                                    rocke_conv_d_addr_pointwise,
                                    (void*)c_K,
                                    d_rsrc,
                                    bound_m,
                                    bound_n,
                                    false);
    }
    else
    {
        /* D_desc = make_d_descriptor(p) */
        rocke_tensor_descriptor_t* D_desc = rocke_conv_make_d_descriptor(b, p);
        rocke_conv_d_addr_ctx_t dctx;
        dctx.D_desc = D_desc;
        dctx.k_out_group_base = rocke_conv_make_k_out_group_base(b, p);
        /* hoist bounds in Python's left-to-right order: M first, then N_gemm */
        rocke_value_t* bound_m = rocke_b_const_i32(b, rocke_conv_problem_m(p));
        rocke_value_t* bound_n = rocke_b_const_i32(b, rocke_conv_problem_n_gemm(p));
        rocke_direct_epilogue_store(b,
                                    &epi,
                                    accs,
                                    num_accs,
                                    rocke_conv_d_addr,
                                    (void*)&dctx,
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
                                          rocke_value_t* c0,
                                          rocke_value_t* ir_c_K_pw)
{
    const rocke_conv_problem_t* p = &spec->problem;
    int mfmas_m = rocke_implicit_gemm_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_implicit_gemm_conv_spec_mfmas_per_warp_n(spec);
    const char* dtype_d = spec->dtype_d;
    bool _fp32_out = dtype_d && dtype_d[0] == 'f' && dtype_d[1] == 'p' && dtype_d[2] == '3'
                     && dtype_d[3] == '2';
    bool _bf16_out = dtype_d && dtype_d[0] == 'b' && dtype_d[1] == 'f' && dtype_d[2] == '1'
                     && dtype_d[3] == '6';
    int elem_bytes = _fp32_out ? 4 : 2;

    /* warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m)) */
    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    /* warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n)) */
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));

    /* c_M = b.const_i32(p.M); c_N = b.const_i32(p.N_gemm) */
    rocke_value_t* c_M = rocke_b_const_i32(b, rocke_conv_problem_m(p));
    rocke_value_t* c_N = rocke_b_const_i32(b, rocke_conv_problem_n_gemm(p));
    /* Pointwise: skip descriptor, use kpg constant for flat D offset. */
    bool _is_pointwise = rocke_conv_problem_is_pointwise(p);
    /* Python _emit_direct_epilogue_wmma emits its own _c_K_wmma = b.const_i32(kpg).
     * Emit a new const here to match the Python SSA sequence. */
    rocke_value_t* c_K_wmma
        = _is_pointwise ? rocke_b_const_i32(b, rocke_conv_problem_kpg(p)) : NULL;
    (void)ir_c_K_pw; /* prologue value not used in wmma path */
    /* D_desc = make_d_descriptor(p) (NULL when pointwise) */
    rocke_tensor_descriptor_t* D_desc = _is_pointwise ? NULL : rocke_conv_make_d_descriptor(b, p);
    /* Grouped conv: k_out_group_base = b.mul(b.block_id_z(), b.const_i32(p.kpg))
     * if p.groups > 1 else None  (Python PR #10064 _emit_direct_epilogue_wmma). */
    rocke_value_t* k_out_group_base = _is_pointwise ? NULL : rocke_conv_make_k_out_group_base(b, p);
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

                /* d_off_elems, _ = D_desc.offset(b, m=m_val, k_out=k_out)
                 * Pointwise fast path: flat offset = m * kpg + n.
                 * Grouped conv: k_out = k_out_group_base + n_val (else n_val). */
                if(_is_pointwise)
                {
                    d_off_elems = rocke_b_add(b, rocke_b_mul(b, m_val, c_K_wmma), n_val);
                }
                else
                {
                    rocke_value_t* k_out = (k_out_group_base != NULL)
                                               ? rocke_b_add(b, k_out_group_base, n_val)
                                               : n_val;
                    const char* names[2];
                    rocke_value_t* values[2];
                    rocke_value_t* valid = NULL;
                    names[0] = "m";
                    names[1] = "k_out";
                    values[0] = m_val;
                    values[1] = k_out;
                    rocke_transforms_descriptor_offset(
                        b, D_desc, names, values, 2, &d_off_elems, &valid);
                }
                d_off_bytes = rocke_b_mul(b, d_off_elems, rocke_b_const_i32(b, elem_bytes));
                safe_off = rocke_b_select(
                    b, ok, d_off_bytes, rocke_b_const_i32(b, (int64_t)((1u << 31) - 1u)));
                if(_fp32_out)
                {
                    rocke_b_buffer_store_f32(b, d_rsrc, safe_off, c0, v_f32);
                }
                else if(_bf16_out)
                {
                    rocke_b_buffer_store_bf16(
                        b, d_rsrc, safe_off, c0, rocke_b_trunc_f32_to_bf16(b, v_f32));
                }
                else
                {
                    rocke_b_buffer_store_f16(
                        b, d_rsrc, safe_off, c0, rocke_b_trunc_f32_to_f16(b, v_f32));
                }
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
                                       rocke_value_t* d_rsrc,
                                       rocke_value_t* ir_c_K_pw,
                                       const rocke_mmaop_t* op)
{
    const rocke_conv_problem_t* p = &spec->problem;
    int max_store_vec;
    if(spec->has_vector_size_c)
    {
        max_store_vec = spec->vector_size_c;
    }
    else
    {
        /* Mirror Python default_vector_sizes(cpg, kpg, dtype_d)[2] (depends only on kpg):
         * largest power-of-two dividing kpg (fp32: max 4, otherwise max 8). */
        int kpg = rocke_conv_problem_kpg(p);
        bool is_fp32_d = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
        if(is_fp32_d)
            max_store_vec = (kpg % 4 == 0) ? 4 : (kpg % 2 == 0) ? 2 : 1;
        else
            max_store_vec = (kpg % 8 == 0) ? 8 : (kpg % 4 == 0) ? 4 : (kpg % 2 == 0) ? 2 : 1;
    }
    int _war_barriers = (spec->pipeline && strcmp(spec->pipeline, "wavelet") == 0) ? 2 : 1;
    rocke_cshuffle_epilogue_t epi;
    if(op != NULL && op->family != NULL && strcmp(op->family, "wmma") == 0)
    {
        epi = rocke_cshuffle_epilogue_from_grid_op(op, grid, max_store_vec);
    }
    else
    {
        const rocke_mfma_atom_t* atom
            = rocke_mfma_atom("f16", spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
        epi = rocke_cshuffle_epilogue_from_grid(atom, grid, max_store_vec);
    }
    epi.out_dtype = spec->dtype_d;
    epi.no_alias = spec->cshuffle_no_alias;
    epi.war_barriers = _war_barriers;

    if(rocke_conv_problem_is_pointwise(p))
    {
        /* Python _emit_cshuffle_epilogue emits _c_K_ir FIRST, then bounds:
         *   _c_K_ir = b.const_i32(kpg)            <- first
         *   bounds = (b.const_i32(M), b.const_i32(N_gemm))   <- second/third */
        rocke_value_t* c_K = rocke_b_const_i32(b, rocke_conv_problem_kpg(p));
        rocke_value_t* bound_m = rocke_b_const_i32(b, rocke_conv_problem_m(p));
        rocke_value_t* bound_n = rocke_b_const_i32(b, rocke_conv_problem_n_gemm(p));
        (void)ir_c_K_pw;
        rocke_cshuffle_epilogue_store(b,
                                      &epi,
                                      accs,
                                      num_accs,
                                      rocke_conv_d_addr_pointwise,
                                      (void*)c_K,
                                      d_rsrc,
                                      bound_m,
                                      bound_n);
    }
    else
    {
        /* D_desc = make_d_descriptor(p) */
        rocke_tensor_descriptor_t* D_desc = rocke_conv_make_d_descriptor(b, p);
        rocke_conv_d_addr_ctx_t dctx;
        dctx.D_desc = D_desc;
        dctx.k_out_group_base = rocke_conv_make_k_out_group_base(b, p);
        /* hoist bounds in Python's left-to-right order: M first, then N_gemm */
        rocke_value_t* bound_m = rocke_b_const_i32(b, rocke_conv_problem_m(p));
        rocke_value_t* bound_n = rocke_b_const_i32(b, rocke_conv_problem_n_gemm(p));
        rocke_cshuffle_epilogue_store(
            b, &epi, accs, num_accs, rocke_conv_d_addr, (void*)&dctx, d_rsrc, bound_m, bound_n);
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
        rocke_conv_emit_cshuffle_epilogue(
            b, spec, final_accs, num_accs, &ctx->grid, ctx->d_rsrc, ctx->ir_c_K_pw, ctx->op);
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
                                             ctx->c0,
                                             ctx->ir_c_K_pw);
    }
    /* else: */
    else
    {
        rocke_conv_emit_direct_epilogue(
            b, spec, final_accs, num_accs, &ctx->grid, ctx->d_rsrc, ctx->ir_c_K_pw);
    }
}
