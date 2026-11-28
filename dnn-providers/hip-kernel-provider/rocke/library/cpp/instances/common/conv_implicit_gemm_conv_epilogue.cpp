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

#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_pack_f32_to */
#include "rocke/instance_conv_implicit_gemm_internal.h"
#include <stdio.h>

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
        /* count is the per-lane MMA result width (4/8/16);
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
    rocke_value_t* slab_base; /* split-K workspace slab base; NULL => store into D */
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
    /* Split-K: the slab is a packed N[D]HWK tensor like D; only its base is
     * added (Python wraps d_addr: b.add(_slab_base, off)). */
    if(dctx->slab_base != NULL)
        off = rocke_b_add(b, dctx->slab_base, off);
    if(out_valid != NULL)
        *out_valid = valid;
    return off;
}

/* Pointwise D-address closure: flat offset = m * K + k_out, always valid.
 * Python: def d_addr(b_, m_val, n_val):
 *             if grouped:
 *                 k_out = b_.add(k_out_group_base, n_val)
 *                 return b_.add(b_.mul(m_val, p_K), k_out), _always_valid_d
 *             return b_.add(b_.mul(m_val, p_kpg), n_val), _always_valid_d
 * where _always_valid_d is ONE const_i32(1) hoisted before the K-loop; every
 * store reuses it, so a fresh constant per call would shift every SSA id. */
typedef struct rocke_conv_d_addr_pw_ctx
{
    rocke_value_t* pitch; /* p_kpg ungrouped (== K), p_K grouped */
    rocke_value_t* k_out_group_base; /* g*kpg, NULL when ungrouped */
    rocke_value_t* valid; /* the hoisted _always_valid_d */
} rocke_conv_d_addr_pw_ctx_t;

/* The pointwise D offset: m * pitch + (k_out_group_base + n) grouped, else
 * m * pitch + n. Python emits the group add before the row multiply. */
static rocke_value_t* rocke_conv_d_off_pointwise(rocke_ir_builder_t* b,
                                                 rocke_value_t* m_val,
                                                 rocke_value_t* n_val,
                                                 rocke_value_t* pitch,
                                                 rocke_value_t* k_out_group_base)
{
    rocke_value_t* k_out
        = (k_out_group_base != NULL) ? rocke_b_add(b, k_out_group_base, n_val) : n_val;
    return rocke_b_add(b, rocke_b_mul(b, m_val, pitch), k_out);
}

static rocke_value_t* rocke_conv_d_addr_pointwise(rocke_ir_builder_t* b,
                                                  rocke_value_t* m_global,
                                                  rocke_value_t* n_global,
                                                  rocke_value_t** out_valid,
                                                  void* user)
{
    const rocke_conv_d_addr_pw_ctx_t* pw = (const rocke_conv_d_addr_pw_ctx_t*)user;
    rocke_value_t* off
        = rocke_conv_d_off_pointwise(b, m_global, n_global, pw->pitch, pw->k_out_group_base);
    if(out_valid != NULL)
        *out_valid = pw->valid;
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
                                     rocke_value_t* ir_d_pitch_pw,
                                     rocke_value_t* always_valid_d,
                                     const rocke_tensor_descriptor_t* D_desc,
                                     rocke_value_t* k_out_group_base,
                                     rocke_value_t* slab_base,
                                     rocke_value_t* bound_m,
                                     rocke_value_t* bound_n)
{
    const rocke_conv_problem_t* p = &spec->problem;
    /* Under group_merge the N coord *is* the merged group index and every one
     * of its Gm values is a real output at a consecutive NHWC channel -- there
     * is no diagonal to gather here, because the mask was spent on the B load.
     * The caller passes the merged group base and N bound; the store itself is
     * untouched and dense. */
    rocke_direct_epilogue_t epi;

    /* atom=spec.atom: keyed on the MMA dtype (the TF32 atom under TF32). */
    epi.atom = rocke_mfma_atom(rocke_conv_data_mma_operand(spec->dtype_a, spec->mma_dtype),
                               spec->warp_tile_m,
                               spec->warp_tile_n,
                               spec->warp_tile_k);
    epi.grid = *grid;
    epi.out_dtype = spec->dtype_d;

    /* AOT, as the cshuffle epilogue: the D descriptor, the grouped k_out base
     * and the store bounds (p_M, p_kpg) are the kernel's runtime values, built
     * once by the caller -- never constants folded from the probe problem.
     * Python: DirectEpilogue(...).store(b, accs=..., addr_fn=d_addr,
     *         d_rsrc=d_rsrc, bounds=(p_M, p_kpg)). */
    if(rocke_conv_problem_is_pointwise(p))
    {
        rocke_conv_d_addr_pw_ctx_t pw;
        pw.pitch = ir_d_pitch_pw;
        pw.k_out_group_base = k_out_group_base;
        pw.valid = always_valid_d;
        rocke_direct_epilogue_store(b,
                                    &epi,
                                    accs,
                                    num_accs,
                                    rocke_conv_d_addr_pointwise,
                                    (void*)&pw,
                                    d_rsrc,
                                    bound_m,
                                    bound_n,
                                    false);
    }
    else
    {
        rocke_conv_d_addr_ctx_t dctx;
        dctx.D_desc = D_desc;
        dctx.k_out_group_base = k_out_group_base;
        dctx.slab_base = slab_base; /* split-K: the slice's fp32 slab */
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
 * Per-lane fp16 store for the WMMA (gfx1151) result layout. The (row, col)
 * of every per-lane slot comes from the op's result layout map
 * (op.dst_layout()) rather than the MFMA-specific MfmaAtom.lane_to_output. Each
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
                                          rocke_value_t* ir_d_pitch_pw,
                                          const rocke_tensor_descriptor_t* D_desc,
                                          rocke_value_t* k_out_group_base,
                                          rocke_value_t* slab_base,
                                          rocke_value_t* bound_m,
                                          rocke_value_t* bound_n)
{
    const rocke_conv_problem_t* p = &spec->problem;
    /* Under group_merge the caller passes the merged N bound (Gm) as bound_n
     * and the merged group's k_out base, as for the other epilogues. */
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

    /* AOT: the store bounds are the p_M / p_kpg kernargs and the D descriptor
     * and grouped k_out base were built before the K-loop, so this epilogue
     * emits no shape constants of its own. */
    rocke_value_t* c_M = bound_m;
    rocke_value_t* c_N = bound_n;
    bool _is_pointwise = rocke_conv_problem_is_pointwise(p);
    /* Pointwise: no descriptor; the flat D offset uses the D row pitch. */
    rocke_value_t* c_K_wmma = _is_pointwise ? ir_d_pitch_pw : NULL;
    /* c_map = op.dst_layout() */
    const rocke_arch_layout_map_t* c_map = rocke_mma_op_dst_layout(op, b);

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

            for(i = 0; i < op->dst.frag_len; ++i)
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
                 * Pointwise fast path: flat offset = m * pitch + k_out.
                 * Grouped conv: k_out = k_out_group_base + n_val (else n_val). */
                if(_is_pointwise)
                {
                    d_off_elems
                        = rocke_conv_d_off_pointwise(b, m_val, n_val, c_K_wmma, k_out_group_base);
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
                    /* Split-K: offset into this slice's workspace slab. */
                    if(slab_base != NULL)
                        d_off_elems = rocke_b_add(b, slab_base, d_off_elems);
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
/* _det_store_slab's ws_addr: D's packed [M, K] layout (a group's channel
 * slab at group * kpg), one slab per slot; always valid. */
typedef struct
{
    rocke_value_t* slab;
    rocke_value_t* p_K;
    rocke_value_t* k_out_group_base;
} conv_ws_addr_ctx_t;

static rocke_value_t* conv_ws_addr(rocke_ir_builder_t* b,
                                   rocke_value_t* m_val,
                                   rocke_value_t* n_val,
                                   rocke_value_t** out_valid,
                                   void* user)
{
    conv_ws_addr_ctx_t* w = (conv_ws_addr_ctx_t*)user;
    if(w->k_out_group_base != NULL)
        n_val = rocke_b_add(b, w->k_out_group_base, n_val);
    rocke_value_t* mk = rocke_b_mul(b, m_val, w->p_K);
    *out_valid = NULL;
    return rocke_b_add(b, w->slab, rocke_b_add(b, mk, n_val));
}

/* Python _atomic() of the wave64 atomic split-K forward: the plain cshuffle
 * atomic store into D, or (split_det 1/2) the deterministic owner fold. */
static void conv_fwd_split_store(rocke_conv_build_ctx_t* ctx,
                                 const rocke_cshuffle_epilogue_t* epi,
                                 rocke_value_t* const* accs,
                                 int num_accs,
                                 rocke_value_t* D,
                                 rocke_value_t* row_pitch,
                                 rocke_value_t* bound_m,
                                 rocke_value_t* bound_n,
                                 rocke_value_t* col_offset,
                                 rocke_conv_d_addr_ctx_t* dctx,
                                 rocke_value_t* d_rsrc)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    if(!spec->split_det && spec->split_zero_first)
    {
        /* The tile was zeroed by its slice 0 at kernel start: wait for that,
         * add, and the last adder resets the flag pair. */
        rocke_value_t* c0 = ctx->c0;
        rocke_value_t* c_1 = rocke_b_const_i32(b, 1);
        rocke_value_t* f_idx = ctx->fwd_szf_idx;
        {
            rocke_if_t t0 = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tid, c0));
            rocke_b_region_enter(b, t0.then_region);
            rocke_b_global_flag_wait_eq_ord(b, ctx->fwd_p_sk_flags, f_idx, c_1, NULL, "monotonic");
            rocke_b_fence(b, "agent", "acquire");
            rocke_b_region_leave(b);
        }
        rocke_b_sync(b);
        rocke_cshuffle_epilogue_atomic_store_ex(
            b, epi, accs, num_accs, D, row_pitch, bound_m, bound_n, col_offset);
        {
            rocke_if_t t0 = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tid, c0));
            rocke_b_region_enter(b, t0.then_region);
            rocke_value_t* n_idx = rocke_b_add(b, f_idx, c_1);
            rocke_value_t* old
                = rocke_b_global_atomic_add(b, ctx->fwd_p_sk_flags, n_idx, c_1, NULL);
            rocke_value_t* last = rocke_b_cmp_eq(b, old, rocke_b_sub(b, ctx->fwd_p_ks_count, c_1));
            rocke_if_t t1 = rocke_b_scf_if(b, last);
            rocke_b_region_enter(b, t1.then_region);
            rocke_b_global_flag_store(b, ctx->fwd_p_sk_flags, n_idx, c0, NULL, "monotonic");
            rocke_b_global_flag_store(b, ctx->fwd_p_sk_flags, f_idx, c0, NULL, "monotonic");
            rocke_b_region_leave(b);
            rocke_b_region_leave(b);
        }
        return;
    }
    if(!spec->split_det && spec->split_flag_store)
    {
        /* Slice 0 stores and raises the flag; the rest wait, add. */
        rocke_value_t* c0 = ctx->c0;
        rocke_value_t* c_1 = rocke_b_const_i32(b, 1);
        rocke_value_t* mt = rocke_b_div(b, ctx->block_m_off_v, rocke_b_const_i32(b, ctx->block_m));
        rocke_value_t* mtn = rocke_b_mul(b, mt, ctx->p_num_pid_n);
        rocke_value_t* nt = rocke_b_div(b, ctx->block_n_off_v, rocke_b_const_i32(b, ctx->block_n));
        rocke_value_t* t_lin = rocke_b_add(b, mtn, nt);
        if(ctx->group_idx != NULL)
        {
            rocke_value_t* tiles = rocke_b_mul(b, ctx->p_num_pid_m, ctx->p_num_pid_n);
            t_lin = rocke_b_add(b, t_lin, rocke_b_mul(b, ctx->group_idx, tiles));
        }
        rocke_value_t* f_idx
            = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, t_lin, rocke_b_const_i32(b, 2)));
        rocke_value_t* n_idx = rocke_b_add(b, f_idx, c_1);
        rocke_if_else_t ie = rocke_b_scf_if_else(b, rocke_b_cmp_eq(b, ctx->fwd_slice_id, c0));
        rocke_b_region_enter(b, ie.then_region);
        {
            rocke_cshuffle_epilogue_store(
                b, epi, accs, num_accs, rocke_conv_d_addr, (void*)dctx, d_rsrc, bound_m, bound_n);
            rocke_b_fence(b, "agent", "release");
            rocke_b_sync(b);
            rocke_if_t t0 = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tid, c0));
            rocke_b_region_enter(b, t0.then_region);
            rocke_b_global_flag_store(b, ctx->fwd_p_sk_flags, f_idx, c_1, "agent", "release");
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        {
            rocke_if_t t0 = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tid, c0));
            rocke_b_region_enter(b, t0.then_region);
            rocke_b_global_flag_wait_eq_ord(b, ctx->fwd_p_sk_flags, f_idx, c_1, NULL, "monotonic");
            rocke_b_fence(b, "agent", "acquire");
            rocke_value_t* old
                = rocke_b_global_atomic_add(b, ctx->fwd_p_sk_flags, n_idx, c_1, NULL);
            rocke_value_t* k2 = rocke_b_sub(b, ctx->fwd_p_ks_count, rocke_b_const_i32(b, 2));
            rocke_if_t t1 = rocke_b_scf_if(b, rocke_b_cmp_eq(b, old, k2));
            rocke_b_region_enter(b, t1.then_region);
            rocke_b_global_flag_store(b, ctx->fwd_p_sk_flags, n_idx, c0, NULL, "monotonic");
            rocke_b_global_flag_store(b, ctx->fwd_p_sk_flags, f_idx, c0, NULL, "monotonic");
            rocke_b_region_leave(b);
            rocke_b_region_leave(b);
            rocke_b_sync(b);
            rocke_cshuffle_epilogue_atomic_store_ex(
                b, epi, accs, num_accs, D, row_pitch, bound_m, bound_n, col_offset);
        }
        rocke_b_region_leave(b);
        return;
    }
    if(!spec->split_det)
    {
        /* Each K slice adds its partial tile into the zeroed D. */
        rocke_cshuffle_epilogue_atomic_store_ex(
            b, epi, accs, num_accs, D, row_pitch, bound_m, bound_n, col_offset);
        return;
    }
    if(spec->split_det == 3)
    {
        /* _det_store_slab: every contribution stores its f32 partial tile,
         * row-major through the cshuffle LDS staging, into its own [M, K]
         * slab of sk_partials (slot = slice); a separate fold kernel sums the
         * slabs in slot order into D. */
        rocke_value_t* slot = ctx->fwd_slice_id;
        if(spec->split_hybrid)
            slot = rocke_b_select(b, ctx->shy_prim, ctx->shy_s, ctx->shy_S);
        rocke_value_t* mk = rocke_b_mul(b, ctx->p_M, ctx->p_K);
        rocke_value_t* slab = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, slot, mk));
        rocke_value_t* ws_rsrc
            = rocke_b_buffer_rsrc(b, ctx->fwd_p_sk_partials, rocke_b_const_i32(b, 0x7FFFFFF0));
        /* Row slabs of <= 32 KB of f32 where the warp's M atoms allow. */
        const int mch = rocke_conv_fwd_split_det_slab_m_chunks(spec);
        rocke_cshuffle_epilogue_t e32 = rocke_cshuffle_epilogue_from_grid(epi->atom, &epi->grid, 4);
        e32.out_dtype = "fp32";
        e32.no_alias = spec->cshuffle_no_alias;
        e32.war_barriers = 1;
        if(mch > 1)
            e32.m_chunks = mch;
        e32.lds_barriers = spec->epi_lds_barriers;
        conv_ws_addr_ctx_t wctx;
        wctx.slab = slab;
        wctx.p_K = ctx->p_K;
        wctx.k_out_group_base = col_offset;
        rocke_cshuffle_epilogue_store(
            b, &e32, accs, num_accs, conv_ws_addr, (void*)&wctx, ws_rsrc, bound_m, bound_n);
        return;
    }
    /* _det_store (modes 1 / 2): the owner (last slice) folds the other
     * contributions' f32 partials in slot order and stores D once; the others
     * store their partial into their slot and count in. Mode 2 moves the
     * partials through agent-coherent (SC1) buffer accesses. */
    rocke_value_t* c0 = ctx->c0;
    const bool coh = spec->split_det == 2;
    const int aux = 16; /* SC1 */
    rocke_value_t* ws_rsrc = NULL;
    if(coh)
        ws_rsrc = rocke_b_buffer_rsrc(b, ctx->fwd_p_sk_partials, rocke_b_const_i32(b, 0x7FFFFFF0));
    rocke_value_t* c_1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_tile = rocke_b_const_i32(b, ctx->block_m * ctx->block_n);
    rocke_value_t *f_idx, *t_slot, *n_wait, *is_owner, *n_slots, *slot;
    if(spec->split_hybrid)
    {
        rocke_value_t* has_rem = rocke_b_cmp_lt(b, c0, ctx->shy_H);
        rocke_value_t* np_ = rocke_b_lnot(b, ctx->shy_prim);
        rocke_value_t* sm1 = rocke_b_sub(b, ctx->shy_S, c_1);
        rocke_value_t* last = rocke_b_land(b, ctx->shy_prim, rocke_b_cmp_eq(b, ctx->shy_s, sm1));
        is_owner = rocke_b_select(b, has_rem, np_, last);
        rocke_value_t* sm1b = rocke_b_sub(b, ctx->shy_S, c_1);
        n_wait = rocke_b_to_sgpr_u32(b, rocke_b_select(b, has_rem, ctx->shy_S, sm1b));
        n_slots = ctx->shy_S;
        slot = ctx->shy_s;
        f_idx = ctx->fsk_tile;
        t_slot = ctx->fsk_tile;
    }
    else
    {
        rocke_value_t* mt = rocke_b_div(b, ctx->block_m_off_v, rocke_b_const_i32(b, ctx->block_m));
        rocke_value_t* mtn = rocke_b_mul(b, mt, ctx->p_num_pid_n);
        rocke_value_t* nt = rocke_b_div(b, ctx->block_n_off_v, rocke_b_const_i32(b, ctx->block_n));
        rocke_value_t* t_lin = rocke_b_add(b, mtn, nt);
        rocke_value_t* tiles = rocke_b_mul(b, ctx->p_num_pid_m, ctx->p_num_pid_n);
        f_idx = t_lin;
        t_slot = t_lin;
        rocke_value_t* per_group = tiles;
        if(spec->split_tail)
        {
            /* Slots for the tail tiles only (launch order). */
            t_slot = rocke_b_sub(b, t_lin, ctx->fwd_p_full_tiles);
            per_group = rocke_b_sub(b, tiles, ctx->fwd_p_full_tiles);
        }
        if(ctx->group_idx != NULL)
        {
            f_idx = rocke_b_add(b, f_idx, rocke_b_mul(b, ctx->group_idx, tiles));
            t_slot = rocke_b_add(b, t_slot, rocke_b_mul(b, ctx->group_idx, per_group));
        }
        n_wait = rocke_b_to_sgpr_u32(b, rocke_b_sub(b, ctx->fwd_p_ks_count, c_1));
        is_owner = rocke_b_cmp_eq(b, ctx->fwd_slice_id, n_wait);
        n_slots = n_wait;
        slot = ctx->fwd_slice_id;
    }
    f_idx = rocke_b_to_sgpr_u32(b, f_idx);
    rocke_value_t* t_base
        = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_mul(b, t_slot, n_slots), c_tile));
    const int bs = rocke_implicit_gemm_conv_spec_block_size(spec);
    (void)rocke_b_const_i32(b, bs); /* c_bs */
    /* Each accumulator vector as 4-wide f32 pieces. */
    int piece_ai[256], piece_q[256], n_pieces = 0;
    for(int ai = 0; ai < num_accs; ++ai)
    {
        const int n = ctx->c_per_lane;
        for(int q = 0; q < n / 4; ++q)
        {
            piece_ai[n_pieces] = ai;
            piece_q[n_pieces++] = q;
        }
    }
    auto piece_off = [&](rocke_value_t* base, int j) -> rocke_value_t* {
        rocke_value_t* cj = rocke_b_const_i32(b, j * bs);
        rocke_value_t* t = rocke_b_add(b, cj, ctx->tid);
        rocke_value_t* m4 = rocke_b_mul(b, t, rocke_b_const_i32(b, 4));
        return rocke_b_add(b, base, m4);
    };
    {
        rocke_if_t nf = rocke_b_scf_if(b, rocke_b_lnot(b, is_owner));
        rocke_b_region_enter(b, nf.then_region);
        rocke_value_t* s_base = rocke_b_add(b, t_base, rocke_b_mul(b, slot, c_tile));
        for(int j = 0; j < n_pieces; ++j)
        {
            rocke_value_t* acc = accs[piece_ai[j]];
            rocke_value_t* v;
            if(ctx->c_per_lane == 4)
                v = acc;
            else
            {
                rocke_value_t* el[4];
                for(int e = 0; e < 4; ++e)
                    el[e] = rocke_b_vec_extract(b, acc, 4 * piece_q[j] + e);
                v = rocke_b_vec_pack(b, el, 4, rocke_f32());
            }
            if(coh)
            {
                rocke_value_t* po = piece_off(s_base, j);
                rocke_value_t* bo = rocke_b_mul(b, po, rocke_b_const_i32(b, 4));
                rocke_b_buffer_store_vN_f32_aux(b, ws_rsrc, bo, c0, v, 4, aux);
            }
            else
                rocke_b_global_store_vN(b, ctx->fwd_p_sk_partials, piece_off(s_base, j), v, 4, 0);
        }
        if(!coh)
            rocke_b_fence(b, "agent", "release");
        rocke_b_sync(b);
        rocke_if_t t0 = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tid, c0));
        rocke_b_region_enter(b, t0.then_region);
        (void)rocke_b_global_atomic_add(b, ctx->fwd_p_sk_flags, f_idx, c_1, NULL);
        rocke_b_region_leave(b);
        rocke_b_region_leave(b);
    }
    rocke_if_t ow = rocke_b_scf_if(b, is_owner);
    rocke_b_region_enter(b, ow.then_region);
    {
        rocke_if_t t0 = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tid, c0));
        rocke_b_region_enter(b, t0.then_region);
        rocke_b_global_flag_wait_eq_ord(b, ctx->fwd_p_sk_flags, f_idx, n_wait, NULL, "monotonic");
        if(!coh)
            rocke_b_fence(b, "agent", "acquire");
        /* Every contribution has counted in: reset for the next launch. */
        rocke_b_global_flag_store(b, ctx->fwd_p_sk_flags, f_idx, c0, NULL, "monotonic");
        rocke_b_region_leave(b);
        rocke_b_sync(b);
        if(!coh)
            rocke_b_fence(b, "agent", "acquire");
        else
            rocke_b_fence(b, "workgroup", "acquire"); /* no partial load above the wait */
        rocke_iter_arg_t it[ROCKE_CONV_MAX_ACCS];
        char nb[32];
        for(int i = 0; i < num_accs; ++i)
        {
            snprintf(nb, sizeof(nb), "sdet_a%d", i);
            it[i].name = rocke_arena_strdup(&b->arena, nb);
            it[i].init = accs[i];
        }
        rocke_for_t fold
            = rocke_b_scf_for_iter(b, c0, n_wait, c_1, it, num_accs, "sdet_s", false, true);
        rocke_b_region_enter(b, fold.body);
        {
            rocke_value_t* s_base = rocke_b_add(b, t_base, rocke_b_mul(b, fold.iv, c_tile));
            rocke_value_t* loaded[256];
            for(int j = 0; j < n_pieces; ++j)
            {
                if(coh)
                {
                    rocke_value_t* po = piece_off(s_base, j);
                    rocke_value_t* bo = rocke_b_mul(b, po, rocke_b_const_i32(b, 4));
                    loaded[j] = rocke_b_buffer_load_vN_aux(b, ws_rsrc, bo, c0, rocke_f32(), 4, aux);
                }
                else
                    loaded[j] = rocke_b_global_load_vN(
                        b, ctx->fwd_p_sk_partials, piece_off(s_base, j), rocke_f32(), 4, 0);
            }
            rocke_value_t* nw[ROCKE_CONV_MAX_ACCS];
            int j = 0;
            for(int ai = 0; ai < num_accs; ++ai)
            {
                rocke_value_t* acc = fold.iter_vars[ai];
                const int n = ctx->c_per_lane;
                if(n == 4)
                {
                    nw[ai] = rocke_b_fadd(b, acc, loaded[j]);
                    ++j;
                    continue;
                }
                rocke_value_t* parts[64];
                int np = 0;
                for(int q = 0; q < n / 4; ++q)
                    for(int e = 0; e < 4; ++e)
                        parts[np++] = rocke_b_vec_extract(b, loaded[j + q], e);
                j += n / 4;
                nw[ai] = rocke_b_fadd(b, acc, rocke_b_vec_pack(b, parts, np, rocke_f32()));
            }
            rocke_b_scf_yield(b, nw, num_accs);
        }
        rocke_b_region_leave(b);
        rocke_cshuffle_epilogue_store(b,
                                      epi,
                                      fold.op->results,
                                      num_accs,
                                      rocke_conv_d_addr,
                                      (void*)dctx,
                                      d_rsrc,
                                      bound_m,
                                      bound_n);
    }
    rocke_b_region_leave(b);
}

/* split_zero_first (Python _szf_idx block, before the K loops): slice 0 of a
 * split tile zeroes the tile's D (bounds-checked wide stores), releases it
 * and raises the flag the slices' atomic stores wait for. */
void rocke_conv_emit_fwd_szf_zero(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* mt = rocke_b_div(b, ctx->block_m_off_v, rocke_b_const_i32(b, ctx->block_m));
    rocke_value_t* mtn = rocke_b_mul(b, mt, ctx->p_num_pid_n);
    rocke_value_t* nt = rocke_b_div(b, ctx->block_n_off_v, rocke_b_const_i32(b, ctx->block_n));
    rocke_value_t* t_lin = rocke_b_add(b, mtn, nt);
    if(ctx->group_idx != NULL)
    {
        rocke_value_t* tiles = rocke_b_mul(b, ctx->p_num_pid_m, ctx->p_num_pid_n);
        t_lin = rocke_b_add(b, t_lin, rocke_b_mul(b, ctx->group_idx, tiles));
    }
    ctx->fwd_szf_idx = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, t_lin, rocke_b_const_i32(b, 2)));
    rocke_value_t* zc = rocke_b_cmp_eq(b, ctx->fwd_slice_id, c0);
    if(spec->split_tail)
        zc = rocke_b_land(b, zc, rocke_b_lnot(b, ctx->fwd_is_full));
    rocke_if_t zi = rocke_b_scf_if(b, zc);
    rocke_b_region_enter(b, zi.then_region);
    {
        const int zsv = 8;
        const int zvpr = ctx->block_n / zsv;
        const int zthreads = ctx->threads;
        rocke_value_t* zbf = rocke_b_vec_trunc_f32_to_bf16(b, rocke_b_zero_vec_f32(b, zsv));
        rocke_value_t* c_eb = rocke_b_const_i32(b, 2);
        rocke_value_t* oob = rocke_b_const_i32(b, 0x7FFFFFFF);
        rocke_value_t* c_nb = ctx->merged ? ctx->c_gm : ctx->p_kpg;
        rocke_conv_d_addr_ctx_t dctx;
        dctx.D_desc = ctx->D_desc;
        dctx.k_out_group_base = ctx->d_k_out_group_base;
        dctx.slab_base = NULL;
        const int ne = (ctx->block_m * zvpr + zthreads - 1) / zthreads;
        for(int e = 0; e < ne; ++e)
        {
            rocke_value_t* vidx = rocke_b_add(b, rocke_b_const_i32(b, e * zthreads), ctx->tid);
            rocke_value_t* row = rocke_b_div(b, vidx, rocke_b_const_i32(b, zvpr));
            rocke_value_t* cm = rocke_b_mod(b, vidx, rocke_b_const_i32(b, zvpr));
            rocke_value_t* col = rocke_b_mul(b, cm, rocke_b_const_i32(b, zsv));
            rocke_value_t* m_val = rocke_b_add(b, ctx->block_m_off_v, row);
            rocke_value_t* n_val = rocke_b_add(b, ctx->block_n_off_v, col);
            rocke_value_t* r_ok = rocke_b_cmp_lt(b, row, rocke_b_const_i32(b, ctx->block_m));
            rocke_value_t* m_ok = rocke_b_cmp_lt(b, m_val, ctx->p_M);
            rocke_value_t* rm = rocke_b_land(b, r_ok, m_ok);
            rocke_value_t* nz = rocke_b_add(b, n_val, rocke_b_const_i32(b, zsv));
            rocke_value_t* n_ok = rocke_b_cmp_le(b, nz, c_nb);
            rocke_value_t* ok = rocke_b_land(b, rm, n_ok);
            rocke_value_t* valid = NULL;
            rocke_value_t* off = rocke_conv_d_addr(b, m_val, n_val, &valid, &dctx);
            if(valid != NULL)
                ok = rocke_b_land(b, ok, valid);
            rocke_value_t* safe = rocke_b_select(b, ok, rocke_b_mul(b, off, c_eb), oob);
            rocke_b_buffer_store_vN_bf16(b, ctx->d_rsrc, safe, c0, zbf, zsv / 2);
        }
        rocke_b_fence(b, "agent", "release");
        rocke_b_sync(b);
        rocke_if_t t0 = rocke_b_scf_if(b, rocke_b_cmp_eq(b, ctx->tid, c0));
        rocke_b_region_enter(b, t0.then_region);
        rocke_b_global_flag_store(
            b, ctx->fwd_p_sk_flags, ctx->fwd_szf_idx, rocke_b_const_i32(b, 1), "agent", "release");
        rocke_b_region_leave(b);
    }
    rocke_b_region_leave(b);
}

void rocke_conv_emit_cshuffle_epilogue(rocke_ir_builder_t* b,
                                       const rocke_implicit_gemm_conv_spec_t* spec,
                                       rocke_value_t* const* accs,
                                       int num_accs,
                                       const rocke_warp_grid_t* grid,
                                       rocke_value_t* d_rsrc,
                                       rocke_value_t* ir_d_pitch_pw,
                                       rocke_value_t* always_valid_d,
                                       const rocke_mmaop_t* op,
                                       const rocke_tensor_descriptor_t* D_desc,
                                       rocke_value_t* k_out_group_base,
                                       rocke_value_t* slab_base,
                                       const char* out_dtype,
                                       rocke_value_t* bound_m,
                                       rocke_value_t* bound_n,
                                       rocke_value_t* atomic_d,
                                       rocke_value_t* atomic_row_pitch,
                                       rocke_conv_build_ctx_t* sctx)
{
    const rocke_conv_problem_t* p = &spec->problem;
    /* This is the epilogue group_merge actually ships on (the validator rejects
     * `default` once the merged store vector exceeds 1). Merging is a strictly
     * better fit for it than CK's arrangement: because the diagonal mask was
     * spent on the B load, all Gm N-lanes hold real outputs at consecutive
     * stride-1 output channels, so the LDS-staged wide store is fully dense. */
    rocke_conv_problem_t merged;
    rocke_implicit_gemm_conv_spec_merged_problem(spec, &merged);
    int max_store_vec;
    if(spec->has_vector_size_c)
    {
        max_store_vec = spec->vector_size_c;
    }
    else
    {
        /* Mirror Python default_vector_sizes(cpg, kpg, dtype_d)[2] (depends only on kpg):
         * largest power-of-two dividing kpg (fp32: max 4, otherwise max 8).
         *
         * The merged problem here is the store-side half of the win: merged kpg
         * is Gm, so this goes from 1 to min(Gm, 8). It must agree with the
         * identical computation in the validator (the build glue's default
         * epilogue gate), which is why both read the merged problem -- if they
         * diverge the validator admits a spec whose epilogue it would have
         * rejected. */
        int kpg = rocke_conv_problem_kpg(&merged);
        bool is_fp32_d = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
        if(is_fp32_d)
            max_store_vec = (kpg % 4 == 0) ? 4 : (kpg % 2 == 0) ? 2 : 1;
        else
            max_store_vec = (kpg % 8 == 0) ? 8 : (kpg % 4 == 0) ? 4 : (kpg % 2 == 0) ? 2 : 1;
    }
    /* The stored element type: dtype_d, except the split-K binary, which
     * stores fp32 partials. Python's CShuffleEpilogue.from_grid[_op] caps an
     * fp32 store at 4 elements (16 bytes); the C helpers learn out_dtype only
     * after construction, so the cap is applied here. A no-op for a dtype_d
     * fp32 spec, whose store width already derives as <= 4. */
    const char* store_dtype = (out_dtype != NULL) ? out_dtype : spec->dtype_d;
    if(store_dtype != NULL && strcmp(store_dtype, "fp32") == 0 && max_store_vec > 4)
        max_store_vec = 4;
    int _war_barriers = (spec->pipeline && strcmp(spec->pipeline, "wavelet") == 0) ? 2 : 1;
    rocke_cshuffle_epilogue_t epi;
    if(op != NULL && op->family != NULL && strcmp(op->family, "wmma") == 0)
    {
        epi = rocke_cshuffle_epilogue_from_grid_op(op, grid, max_store_vec);
    }
    else
    {
        /* atom=spec.atom: keyed on the MMA dtype (the TF32 atom under TF32). */
        const rocke_mfma_atom_t* atom
            = rocke_mfma_atom(rocke_conv_data_mma_operand(spec->dtype_a, spec->mma_dtype),
                              spec->warp_tile_m,
                              spec->warp_tile_n,
                              spec->warp_tile_k);
        epi = rocke_cshuffle_epilogue_from_grid(atom, grid, max_store_vec);
    }
    /* out_dtype: the stored element type. dtype_d, except the split-K binary,
     * which stores fp32 partials (the store width above still derives from
     * dtype_d, as in Python). */
    epi.out_dtype = store_dtype;
    epi.no_alias = spec->cshuffle_no_alias;
    epi.war_barriers = _war_barriers;
    if(spec->cshuffle_m_chunks > 1)
        epi.m_chunks = spec->cshuffle_m_chunks;
    epi.lds_barriers = spec->epi_lds_barriers;
    if(spec->epi_store_fast && !rocke_conv_problem_is_pointwise(p) && sctx != NULL)
    {
        /* epi_store_fast: packed N[D]HWK and the tile in bounds -> the wide
         * stores at offset m * K + k_out, a constant row step. */
        rocke_conv_build_ctx_t* c = sctx;
        rocke_value_t* e1 = rocke_b_cmp_eq(b, c->p_D_stride_wo, c->p_K);
        rocke_value_t* wk = rocke_b_mul(b, c->p_Wo, c->p_K);
        rocke_value_t* e2 = rocke_b_cmp_eq(b, c->p_D_stride_ho, wk);
        rocke_value_t* pk = rocke_b_land(b, e1, e2);
        if(p->is_3d)
        {
            rocke_value_t* hh = rocke_b_mul(b, c->p_Ho, c->p_D_stride_ho);
            pk = rocke_b_land(b, pk, rocke_b_cmp_eq(b, c->p_D_stride_do, hh));
            rocke_value_t* dd = rocke_b_mul(b, c->p_Do, c->p_D_stride_do);
            pk = rocke_b_land(b, pk, rocke_b_cmp_eq(b, c->p_D_stride_n, dd));
        }
        else
        {
            rocke_value_t* hh = rocke_b_mul(b, c->p_Ho, c->p_D_stride_ho);
            pk = rocke_b_land(b, pk, rocke_b_cmp_eq(b, c->p_D_stride_n, hh));
        }
        rocke_value_t* me = rocke_b_add(b, c->block_m_off_v, rocke_b_const_i32(b, c->block_m));
        pk = rocke_b_land(b, pk, rocke_b_cmp_le(b, me, bound_m));
        rocke_value_t* ne = rocke_b_add(b, c->block_n_off_v, rocke_b_const_i32(b, c->block_n));
        pk = rocke_b_land(b, pk, rocke_b_cmp_le(b, ne, bound_n));
        epi.fast_ok = pk;
        epi.fast_row_pitch = c->p_K;
        epi.fast_col_base = k_out_group_base;
    }

    /* AOT: the D descriptor, the grouped k_out base and the store bounds all
     * come from the build context. Python builds them before the K-loop (so
     * the wavelet driver can reuse the same store code) and bounds with the
     * p_M / p_kpg kernargs, so nothing is emitted here. */
    if(atomic_d != NULL)
    {
        epi.atomic_pair_lanes = spec->split_atomic_pairs;
        if(spec->split_atomic_agent)
            epi.atomic_syncscope = "agent";
        if(spec->split_atomic_fast)
            epi.atomic_fast_full = true;
        rocke_conv_d_addr_ctx_t dctx;
        dctx.D_desc = D_desc;
        dctx.k_out_group_base = k_out_group_base;
        dctx.slab_base = slab_base;
        if(!spec->split_tail)
        {
            conv_fwd_split_store(sctx,
                                 &epi,
                                 accs,
                                 num_accs,
                                 atomic_d,
                                 atomic_row_pitch,
                                 bound_m,
                                 bound_n,
                                 k_out_group_base,
                                 &dctx,
                                 d_rsrc);
            return;
        }
        /* split_tail: a whole-K tile stores plainly; a tail slice adds. */
        rocke_if_else_t ie = rocke_b_scf_if_else(b, sctx->fwd_is_full);
        rocke_b_region_enter(b, ie.then_region);
        rocke_cshuffle_epilogue_store(
            b, &epi, accs, num_accs, rocke_conv_d_addr, (void*)&dctx, d_rsrc, bound_m, bound_n);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        conv_fwd_split_store(sctx,
                             &epi,
                             accs,
                             num_accs,
                             atomic_d,
                             atomic_row_pitch,
                             bound_m,
                             bound_n,
                             k_out_group_base,
                             &dctx,
                             d_rsrc);
        rocke_b_region_leave(b);
        return;
    }
    if(rocke_conv_problem_is_pointwise(p))
    {
        rocke_conv_d_addr_pw_ctx_t pw;
        pw.pitch = ir_d_pitch_pw;
        pw.k_out_group_base = k_out_group_base;
        pw.valid = always_valid_d;
        rocke_cshuffle_epilogue_store(b,
                                      &epi,
                                      accs,
                                      num_accs,
                                      rocke_conv_d_addr_pointwise,
                                      (void*)&pw,
                                      d_rsrc,
                                      bound_m,
                                      bound_n);
    }
    else
    {
        rocke_conv_d_addr_ctx_t dctx;
        dctx.D_desc = D_desc;
        dctx.k_out_group_base = k_out_group_base;
        dctx.slab_base = slab_base;
        rocke_cshuffle_epilogue_store(
            b, &epi, accs, num_accs, rocke_conv_d_addr, (void*)&dctx, d_rsrc, bound_m, bound_n);
    }
}

/* ===================================================================== *
 * rocke_conv_emit_cluster_fold   (Python _emit_cluster_fold)
 *
 * Fold the cluster's S slabs of this tile into D (spec.split_k_cluster).
 * Slice s owns rows [s*R, (s+1)*R) of the tile, R = ceil(TM / S); each
 * thread walks 4-channel quads of those rows (D and the slabs are packed
 * [M][K]; rocke_conv_fwd_split_k_cluster_problem_ok requires K % 4 == 0 and
 * kpg % 4 == 0 at build and launch admission). Every SSA value is
 * bound in Python's left-to-right evaluation order.
 * ===================================================================== */
static void rocke_conv_emit_cluster_fold(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const int S = spec->split_k_cluster;
    const int block_m = ctx->block_m;
    const int block_n = ctx->block_n;
    const int rows = (block_m + S - 1) / S;
    const int quads_per_row = block_n / 4;
    const int n_quads = rows * quads_per_row;
    const int threads = rocke_implicit_gemm_conv_spec_block_size(spec);
    rocke_value_t* m0 = ctx->block_m_off_v;
    rocke_value_t* n0 = ctx->block_n_off_v;
    rocke_value_t* n_bound = ctx->merged ? ctx->c_gm : ctx->p_kpg;
    rocke_value_t* k_base = ctx->k_out_group_base;

    rocke_b_cluster_barrier(b);
    rocke_value_t* c_rows = rocke_b_const_i32(b, rows);
    rocke_value_t* row_step = rocke_b_mul(b, ctx->fwd_slice_id, c_rows);
    rocke_value_t* row0 = rocke_b_add(b, m0, row_step);
    rocke_value_t* slab = rocke_b_mul(b, ctx->p_M, ctx->p_K);
    for(int t = 0; t < (n_quads + threads - 1) / threads; ++t)
    {
        rocke_value_t* c_t = rocke_b_const_i32(b, t * threads);
        rocke_value_t* q = rocke_b_add(b, ctx->tid, c_t);
        rocke_value_t* c_qpr = rocke_b_const_i32(b, quads_per_row);
        rocke_value_t* r = rocke_b_div(b, q, c_qpr);
        rocke_value_t* c_qpr2 = rocke_b_const_i32(b, quads_per_row);
        rocke_value_t* r_q = rocke_b_mul(b, r, c_qpr2);
        rocke_value_t* q_in_row = rocke_b_sub(b, q, r_q);
        rocke_value_t* c_four = rocke_b_const_i32(b, 4);
        rocke_value_t* c = rocke_b_mul(b, q_in_row, c_four);
        rocke_value_t* m = rocke_b_add(b, row0, r);
        rocke_value_t* n = rocke_b_add(b, n0, c);
        rocke_value_t* c_nq = rocke_b_const_i32(b, n_quads);
        rocke_value_t* q_ok = rocke_b_cmp_lt(b, q, c_nq);
        rocke_value_t* m_ok = rocke_b_cmp_lt(b, m, ctx->p_M);
        rocke_value_t* ok_a = rocke_b_land(b, q_ok, m_ok);
        rocke_value_t* c_bm = rocke_b_const_i32(b, block_m);
        rocke_value_t* m_end = rocke_b_add(b, m0, c_bm);
        rocke_value_t* m_in_tile = rocke_b_cmp_lt(b, m, m_end);
        rocke_value_t* n_ok = rocke_b_cmp_lt(b, n, n_bound);
        rocke_value_t* ok_b = rocke_b_land(b, m_in_tile, n_ok);
        rocke_value_t* ok = rocke_b_land(b, ok_a, ok_b);

        rocke_if_t guard = rocke_b_scf_if(b, ok);
        rocke_b_region_enter(b, guard.then_region);
        {
            rocke_value_t* k_out = (k_base == NULL) ? n : rocke_b_add(b, k_base, n);
            rocke_value_t* m_row = rocke_b_mul(b, m, ctx->p_K);
            rocke_value_t* off = rocke_b_add(b, m_row, k_out);
            /* S <= 15 (rocke_conv_fwd_split_k_available). */
            rocke_value_t* parts[15][4] = {};
            for(int s_i = 0; s_i < S; ++s_i)
            {
                rocke_value_t* o = off;
                if(s_i != 0)
                {
                    rocke_value_t* c_s = rocke_b_const_i32(b, s_i);
                    rocke_value_t* s_off = rocke_b_mul(b, c_s, slab);
                    o = rocke_b_add(b, s_off, off);
                }
                rocke_value_t* v = rocke_b_global_load_vN(b, ctx->fwd_p_ws, o, rocke_f32(), 4, 0);
                for(int i = 0; i < 4; ++i)
                    parts[s_i][i] = rocke_b_vec_extract(b, v, i);
            }
            rocke_value_t* tot[4];
            for(int i = 0; i < 4; ++i)
                tot[i] = parts[0][i];
            for(int s_i = 1; s_i < S; ++s_i)
            {
                for(int i = 0; i < 4; ++i)
                    tot[i] = rocke_b_fadd(b, tot[i], parts[s_i][i]);
            }
            rocke_value_t* packed = rocke_b_pack_f32_to(b, tot, 4, spec->dtype_d);
            rocke_b_global_store_vN(b, ctx->D, off, packed, 4, 0);
        }
        rocke_b_region_leave(b);
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
/* Python _emit_fwd_epilogue_qx (epi_quad_xpose): register-only store of the
 * 16x16-atom accumulators through a lane transpose (DPP + v_perm). */
static void conv_emit_fwd_epilogue_qx(rocke_conv_build_ctx_t* ctx, rocke_value_t* const* accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const int mfmas_m = ctx->mfmas_m, mfmas_n = ctx->mfmas_n;
    const int eqx = spec->epi_quad_xpose;
    const bool bf = spec->dtype_d != NULL && strcmp(spec->dtype_d, "bf16") == 0;
    const rocke_type_t* dt = bf ? rocke_bf16() : rocke_f16();
    const bool x3 = eqx == 3 || eqx == 4;
    const bool fast_cvt = eqx == 4 && bf;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* lane = ctx->lane;
    rocke_value_t* c1_ = rocke_b_const_i32(b, x3 ? 4 : 1);
    rocke_value_t* c2_ = rocke_b_const_i32(b, x3 ? 8 : 2);
    rocke_value_t* q_hi = rocke_b_cmp_ne(b, rocke_b_land(b, lane, c2_), c0);
    rocke_value_t* q_odd = rocke_b_cmp_ne(b, rocke_b_land(b, lane, c1_), c0);
    rocke_value_t* s0 = rocke_b_const_i32(b, 0x05040100);
    rocke_value_t* sel_s = rocke_b_select(b, q_odd, s0, rocke_b_const_i32(b, 0x07060302));
    rocke_value_t* s1 = rocke_b_const_i32(b, 0x03020504);
    rocke_value_t* sel_o0 = rocke_b_select(b, q_odd, s1, rocke_b_const_i32(b, 0x05040100));
    rocke_value_t* s2 = rocke_b_const_i32(b, 0x03020706);
    rocke_value_t* sel_o1 = rocke_b_select(b, q_odd, s2, rocke_b_const_i32(b, 0x07060100));
    rocke_value_t *row_in, *col_in;
    if(x3)
    {
        rocke_value_t* l16 = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
        rocke_value_t* r4 = rocke_b_mul(b, l16, rocke_b_const_i32(b, 4));
        rocke_value_t* l4 = rocke_b_div(b, lane, rocke_b_const_i32(b, 4));
        row_in = rocke_b_add(b, r4, rocke_b_land(b, l4, rocke_b_const_i32(b, 3)));
        rocke_value_t* l3 = rocke_b_land(b, lane, rocke_b_const_i32(b, 3));
        col_in = rocke_b_mul(b, l3, rocke_b_const_i32(b, 4));
    }
    else
    {
        rocke_value_t* l16 = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
        rocke_value_t* r4 = rocke_b_mul(b, l16, rocke_b_const_i32(b, 4));
        row_in = rocke_b_add(b, r4, rocke_b_land(b, lane, rocke_b_const_i32(b, 3)));
        col_in = rocke_b_land(b, lane, rocke_b_const_i32(b, 12));
    }
    rocke_value_t* wmo = rocke_warp_grid_warp_m_off(b, &ctx->grid);
    rocke_value_t* m_base = rocke_b_add(b, rocke_b_add(b, ctx->block_m_off_v, wmo), row_in);
    rocke_value_t* wno = rocke_warp_grid_warp_n_off(b, &ctx->grid);
    rocke_value_t* n_base = rocke_b_add(b, rocke_b_add(b, ctx->block_n_off_v, wno), col_in);
    rocke_value_t* c_eb_d = rocke_b_const_i32(b, 2);
    rocke_value_t* oob = rocke_b_const_i32(b, 0x7FFFFFFF);
    const rocke_type_t* wt2 = rocke_vector_type(b, rocke_i32(), 2);
    const bool wide = eqx == 2 || eqx == 3 || eqx == 4;
    rocke_value_t* c_lsb_sh = rocke_b_const_i32(b, 16);
    rocke_value_t* c1_one = rocke_b_const_i32(b, 1);
    rocke_value_t* c_7fff = rocke_b_const_i32(b, 0x7FFF);
    rocke_value_t* c_qnan = rocke_b_const_i32(b, 0x7FC00000);
    rocke_value_t* s_hi2 = rocke_b_const_i32(b, 0x07060302);
    auto cvt_pair = [&](rocke_value_t* acc, int i0) -> rocke_value_t* {
        rocke_value_t* ts[2];
        for(int k = 0; k < 2; ++k)
        {
            rocke_value_t* fv = rocke_b_vec_extract(b, acc, i0 + k);
            rocke_value_t* uv = rocke_b_bitcast(b, fv, rocke_i32());
            rocke_value_t* lsb = rocke_b_land(b, rocke_b_lshr(b, uv, c_lsb_sh), c1_one);
            rocke_value_t* t = rocke_b_add(b, rocke_b_add(b, uv, c_7fff), lsb);
            rocke_value_t* nan = rocke_b_fcmp(b, "uno", fv, fv);
            ts[k] = rocke_b_select(b, nan, c_qnan, t);
        }
        return rocke_b_perm_b32(b, ts[1], ts[0], s_hi2);
    };
    if(wide)
    {
        rocke_value_t* wn2 = rocke_warp_grid_warp_n_off(b, &ctx->grid);
        rocke_value_t* bw = rocke_b_add(b, ctx->block_n_off_v, wn2);
        n_base = rocke_b_add(b, bw, rocke_b_add(b, col_in, col_in));
    }
    rocke_value_t* m_str = NULL;
    if(spec->xh_stride_m)
    {
        rocke_value_t* wm2 = rocke_warp_grid_warp_m_off(b, &ctx->grid);
        rocke_value_t* m_wave = rocke_b_add(b, ctx->block_m_off_v, wm2);
        m_str = rocke_b_add(b, m_wave, rocke_b_mul(b, row_in, rocke_b_const_i32(b, mfmas_m)));
    }
    /* d_addr: the pointwise closure (flat [M, K], always valid) or the D
     * descriptor's, as the other fwd epilogues pick it. */
    rocke_conv_d_addr_ctx_t dctx;
    dctx.D_desc = ctx->D_desc;
    dctx.k_out_group_base = ctx->d_k_out_group_base;
    dctx.slab_base = NULL;
    rocke_conv_d_addr_pw_ctx_t pw;
    pw.pitch = ctx->ir_d_pitch_pw;
    pw.k_out_group_base = ctx->d_k_out_group_base;
    pw.valid = ctx->ir_always_valid_d;
    rocke_value_t* n_bound = ctx->merged ? ctx->c_gm : ctx->p_kpg;
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        rocke_value_t* m_val = rocke_b_add(b, m_base, rocke_b_const_i32(b, mi * 16));
        if(spec->xh_stride_m)
        {
            rocke_value_t* ms = rocke_b_add(b, m_str, rocke_b_const_i32(b, mi));
            m_val = rocke_b_select(b, ctx->xsm_ok, ms, m_val);
        }
        rocke_value_t* valid = NULL;
        rocke_value_t* off0 = ctx->is_pointwise
                                  ? rocke_conv_d_addr_pointwise(b, m_val, n_base, &valid, &pw)
                                  : rocke_conv_d_addr(b, m_val, n_base, &valid, &dctx);
        rocke_value_t* m_ok = rocke_b_cmp_lt(b, m_val, ctx->p_M);
        if(valid != NULL)
            m_ok = rocke_b_land(b, m_ok, valid);
        rocke_value_t* pend[2] = {NULL, NULL};
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = accs[mi * mfmas_n + ni];
            rocke_value_t *w0, *w1;
            if(fast_cvt)
            {
                w0 = cvt_pair(acc, 0);
                w1 = cvt_pair(acc, 2);
            }
            else
            {
                rocke_value_t* w = rocke_b_vec_bitcast(b, rocke_b_vec_cast_f32_to(b, acc, dt), wt2);
                w0 = rocke_b_vec_extract(b, w, 0);
                w1 = rocke_b_vec_extract(b, w, 1);
            }
            rocke_value_t* snd = rocke_b_select(b, q_hi, w0, w1);
            rocke_value_t* rcv = rocke_b_dpp_xor(b, snd, x3 ? 8 : 2);
            rocke_value_t* a_ = rocke_b_select(b, q_hi, rcv, w0);
            rocke_value_t* bv = rocke_b_select(b, q_hi, w1, rcv);
            rocke_value_t* pm = rocke_b_perm_b32(b, bv, a_, sel_s);
            rocke_value_t* t_;
            if(!x3)
                t_ = rocke_b_dpp_xor(b, pm, 1);
            else
            {
                rocke_value_t* sr = rocke_b_mov_dpp(b, pm, 4, -1, false);
                rocke_value_t* sl = rocke_b_mov_dpp(b, pm, -1, 4, false);
                t_ = rocke_b_select(b, q_odd, sr, sl);
            }
            rocke_value_t* o0 = rocke_b_perm_b32(b, t_, a_, sel_o0);
            rocke_value_t* o1 = rocke_b_perm_b32(b, t_, bv, sel_o1);
            if(wide && ni % 2 == 0)
            {
                pend[0] = o0;
                pend[1] = o1;
                continue;
            }
            rocke_value_t* words[4];
            int nw = 0;
            if(wide)
            {
                words[nw++] = pend[0];
                words[nw++] = pend[1];
            }
            words[nw++] = o0;
            words[nw++] = o1;
            const int n_rel = wide ? 32 * (ni / 2) : 16 * ni;
            rocke_value_t* n_val = rocke_b_add(b, n_base, rocke_b_const_i32(b, n_rel));
            rocke_value_t* ne = rocke_b_add(b, n_val, rocke_b_const_i32(b, 2 * nw));
            rocke_value_t* ok = rocke_b_land(b, m_ok, rocke_b_cmp_le(b, ne, n_bound));
            rocke_value_t* off = rocke_b_add(b, off0, rocke_b_const_i32(b, n_rel));
            rocke_value_t* safe = rocke_b_select(b, ok, rocke_b_mul(b, off, c_eb_d), oob);
            rocke_value_t* pk = rocke_b_vec_pack(b, words, nw, rocke_i32());
            rocke_value_t* v = rocke_b_vec_bitcast(b, pk, rocke_vector_type(b, dt, 2 * nw));
            if(bf)
                rocke_b_buffer_store_vN_bf16(b, ctx->d_rsrc, safe, c0, v, nw);
            else
                rocke_b_buffer_store_vN_f16(b, ctx->d_rsrc, safe, c0, v, nw);
        }
    }
}

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
    /* epi_quad_xpose: the register-only store (Python _emit_fwd_epilogue_qx). */
    else if(spec->epi_quad_xpose)
    {
        conv_emit_fwd_epilogue_qx(ctx, final_accs);
    }
    /* The store's N bound is Python's c_N_bound: kpg, or kpg*Gm == Gm under
     * merge (the true kpg, 1 on depthwise, would clamp away Gm-1 of every Gm
     * real outputs). */
    /* elif spec.epilogue == "cshuffle": */
    else if(spec->epilogue != NULL && strcmp(spec->epilogue, "cshuffle") == 0)
    {
        rocke_conv_emit_cshuffle_epilogue(b,
                                          spec,
                                          final_accs,
                                          num_accs,
                                          &ctx->grid,
                                          ctx->d_rsrc,
                                          ctx->ir_d_pitch_pw,
                                          ctx->ir_always_valid_d,
                                          ctx->op,
                                          ctx->D_desc,
                                          ctx->d_k_out_group_base,
                                          ctx->fwd_split ? ctx->fwd_slab_base : NULL,
                                          ctx->fwd_split ? "fp32" : NULL,
                                          ctx->p_M,
                                          ctx->merged ? ctx->c_gm : ctx->p_kpg,
                                          ctx->fwd_asplit ? ctx->D : NULL,
                                          ctx->fwd_asplit ? ctx->p_K : NULL,
                                          ctx);
    }
    /* elif op.family == "wmma": */
    else if(ctx->op != NULL && ctx->op->family != NULL && strcmp(ctx->op->family, "wmma") == 0)
    {
        /* Split-K stores fp32 partials: Python hands this epilogue the spec
         * with dtype_d replaced by fp32 (_epi_spec). */
        rocke_implicit_gemm_conv_spec_t epi_spec = *spec;
        if(ctx->fwd_split)
            epi_spec.dtype_d = "fp32";
        rocke_conv_emit_direct_epilogue_wmma(b,
                                             &epi_spec,
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
                                             ctx->ir_d_pitch_pw,
                                             ctx->D_desc,
                                             ctx->d_k_out_group_base,
                                             ctx->fwd_split ? ctx->fwd_slab_base : NULL,
                                             ctx->p_M,
                                             ctx->merged ? ctx->c_gm : ctx->p_kpg);
    }
    /* else: */
    else
    {
        /* Split-K stores fp32 partials into the slice's slab, as on WMMA. */
        rocke_implicit_gemm_conv_spec_t epi_spec = *spec;
        if(ctx->fwd_split)
            epi_spec.dtype_d = "fp32";
        rocke_conv_emit_direct_epilogue(b,
                                        &epi_spec,
                                        final_accs,
                                        num_accs,
                                        &ctx->grid,
                                        ctx->d_rsrc,
                                        ctx->ir_d_pitch_pw,
                                        ctx->ir_always_valid_d,
                                        ctx->D_desc,
                                        ctx->d_k_out_group_base,
                                        ctx->fwd_split ? ctx->fwd_slab_base : NULL,
                                        ctx->p_M,
                                        ctx->merged ? ctx->c_gm : ctx->p_kpg);
    }
    /* Cluster split-K: after the slab store, fold the cluster's slabs into D
     * (Python: only on the stock epilogue path, never under an override). */
    if(!(ctx->ov != NULL && ctx->ov->epilogue_override != NULL) && ctx->fwd_split
       && spec->split_k_cluster)
    {
        rocke_conv_emit_cluster_fold(ctx);
    }
}
