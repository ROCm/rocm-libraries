// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of build_large_n_opt_matmul_nbits from
 * rocke/instances/common/_matmul_nbits_large_n_opt.py.
 *
 * The function below reproduces the Python IRBuilder call sequence op-for-op so
 * the produced rocke_kernel_def_t is byte-faithful to the Python output. Every
 * node is arena-owned (b->arena); nothing is freed individually.
 *
 * Peer helpers (rocke_wmma_params, rocke_matmul_nbits_*, rocke_unpack_i4_byte_to_pair_f16)
 * are declared in their sibling headers and resolved at link time.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.i4_dequant.h"
#include "rocke/helper_rocke.instances.common._matmul_nbits_common.h"
#include "rocke/helper_rocke.instances.common._matmul_nbits_large_n.h"
#include "rocke/helper_rocke.instances.common._matmul_nbits_large_n_opt.h"
#include "rocke/ir.h"

/* getattr(b, wp.wmma_op) -- dispatch the WMMA atom by its IRBuilder method name.
 * Only the two names _wmma_params can ever produce are handled; an unknown name
 * sets the builder error and returns NULL. */
static rocke_value_t* rocke_opt_wmma(rocke_ir_builder_t* b,
                                     const char* wmma_op,
                                     rocke_value_t* a,
                                     rocke_value_t* bb,
                                     rocke_value_t* c)
{
    if(wmma_op != NULL && strcmp(wmma_op, "wmma_gfx12_f32_16x16x16_f16") == 0)
    {
        return rocke_b_wmma_gfx12_f32_16x16x16_f16(b, a, bb, c);
    }
    if(wmma_op != NULL && strcmp(wmma_op, "wmma_f32_16x16x16_f16") == 0)
    {
        return rocke_b_wmma_f32_16x16x16_f16(b, a, bb, c);
    }
    return NULL;
}

rocke_kernel_def_t* rocke_build_large_n_opt_matmul_nbits(rocke_ir_builder_t* b,
                                                         const rocke_matmul_nbits_spec_t* spec,
                                                         const char* arch)
{
    rocke_wmma_params_t wp;
    const rocke_gemm_tile_spec_t* t;
    int N, K, group;
    int wtm, wtn, wtk, wave;
    int rows_per_wave, cols_per_wave, n_sub_m, n_sub_n, n_acc, n_ksub;
    int k_packed_stride, k_group_stride;
    int tile_m, tile_n, tile_k, bs;
    int a_elems, c_elems, a_chunks, c_chunks;
    const char* scale_wire = NULL;
    const rocke_type_t* scale_t;
    const rocke_type_t* F16t;
    const rocke_type_t* I8t;

    if(arch == NULL)
    {
        arch = "gfx1201";
    }

    if(rocke_wmma_params(arch, &wp) != ROCKE_OK)
    {
        return NULL;
    }

    t = &spec->tile;
    N = spec->N;
    K = spec->K;
    group = spec->group_size;

    /* opt body requires tile_k == group_size (Python ValueError). */
    if(t->tile_k != group)
    {
        return NULL;
    }

    wtm = t->warp_tile_m;
    wtn = t->warp_tile_n;
    wtk = t->warp_tile_k;
    wave = spec->wave_size;

    rows_per_wave = t->tile_m / t->warp_m;
    cols_per_wave = t->tile_n / t->warp_n;
    n_sub_m = rows_per_wave / wtm;
    n_sub_n = cols_per_wave / wtn;
    n_acc = n_sub_m * n_sub_n;
    n_ksub = group / wtk; /* WMMA K-substeps inside one group tile */
    k_packed_stride = K / 2;
    k_group_stride = K / group;

    tile_m = t->tile_m;
    tile_n = t->tile_n;
    tile_k = t->tile_k;
    bs = spec->block_size;

    /* Cooperative-load geometry (Python ValueError on indivisibility). */
    a_elems = tile_m * tile_k;
    c_elems = tile_m * tile_n;
    if((a_elems % (bs * 8)) || (c_elems % (bs * 8)))
    {
        return NULL;
    }
    a_chunks = a_elems / (bs * 8);
    c_chunks = c_elems / (bs * 8);

    if(rocke_matmul_nbits_scale_wire_dtype(spec->scale_dtype, &scale_wire) != ROCKE_OK)
    {
        return NULL;
    }
    F16t = rocke_f16();
    I8t = rocke_i8();
    scale_t = (strcmp(scale_wire, "f16") == 0) ? rocke_f16() : rocke_f32();

    /* b = IRBuilder(spec.kernel_name()) is the caller's responsibility (the
     * builder is already init'd); see rocke_build_large_n_opt_matmul_nbits_new. */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", bs);

    /* --- kernel params --- */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* Sp;
    rocke_value_t* C;
    /* rocke_value_t* M kept for ABI parity (full-tile precondition; unused). */
    {
        rocke_param_opts_t opts;

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = rocke_b_param(b, "A", rocke_ptr_type(b, F16t, "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        Bp = rocke_b_param(b, "B", rocke_ptr_type(b, I8t, "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 8;
        opts.align_set = true;
        Sp = rocke_b_param(b, "Scales", rocke_ptr_type(b, scale_t, "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        C = rocke_b_param(b, "C", rocke_ptr_type(b, F16t, "global"), &opts);

        (void)rocke_b_param(b, "M", rocke_i32(), NULL);
    }

    /* --- LDS staging buffers --- */
    int a_shape[2];
    int c_shape[2];
    rocke_value_t* a_smem;
    rocke_value_t* c_smem;
    a_shape[0] = tile_m;
    a_shape[1] = tile_k;
    c_shape[0] = tile_m;
    c_shape[1] = tile_n;
    a_smem = rocke_b_smem_alloc(b, F16t, a_shape, 2, "A_smem");
    c_smem = rocke_b_smem_alloc(b, F16t, c_shape, 2, "C_smem");

    /* --- constants --- */
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* cK = rocke_b_const_i32(b, K);
    rocke_value_t* cN = rocke_b_const_i32(b, N);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    /* c8 is emitted by the Python (const_i32(8)) and, although never referenced,
     * each const_i32 emits a fresh arith.constant op (no caching) -- so it MUST
     * be emitted here to keep the SSA op sequence byte-identical. */
    rocke_value_t* c8 = rocke_b_const_i32(b, 8);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* c_tile_k = rocke_b_const_i32(b, tile_k);
    rocke_value_t* c_group = rocke_b_const_i32(b, group);
    rocke_value_t* cwave = rocke_b_const_i32(b, wave);
    (void)c8; /* deliberately unused (see above); kept for IR fidelity. */

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* wave_id = rocke_b_div(b, tid, cwave);
    rocke_value_t* lane = rocke_b_mod(b, tid, cwave);
    rocke_value_t* frag = rocke_b_mod(b, lane, c16); /* lane%16 */
    rocke_value_t* half = rocke_b_div(b, lane, c16); /* lane/16 */

    rocke_value_t* half_k_elem
        = wp.split_k_by_half ? rocke_b_mul(b, half, rocke_b_const_i32(b, wp.frag_k)) : c0;
    rocke_value_t* half_k_byte
        = wp.split_k_by_half ? rocke_b_mul(b, half, rocke_b_const_i32(b, wp.frag_k / 2)) : c0;

    rocke_value_t* wave_m = rocke_b_div(b, wave_id, rocke_b_const_i32(b, t->warp_n));
    rocke_value_t* wave_n = rocke_b_mod(b, wave_id, rocke_b_const_i32(b, t->warp_n));

    /* Python evaluates b.block_id_y() (then b.const_i32(tile_m)) left-to-right;
     * C argument evaluation order is unspecified, so force the block-id intrinsic
     * to be emitted FIRST to keep the SSA value numbering byte-identical. */
    rocke_value_t* bidy = rocke_b_block_id_y(b);
    rocke_value_t* m0 = rocke_b_mul(b, bidy, rocke_b_const_i32(b, tile_m));
    rocke_value_t* bidx = rocke_b_block_id_x(b);
    rocke_value_t* n0 = rocke_b_mul(b, bidx, rocke_b_const_i32(b, tile_n));
    rocke_value_t* wm_local = rocke_b_mul(b, wave_m, rocke_b_const_i32(b, rows_per_wave));
    rocke_value_t* wn_base
        = rocke_b_add(b, n0, rocke_b_mul(b, wave_n, rocke_b_const_i32(b, cols_per_wave)));

    /* Loop-invariant per-lane B row bases (packed-byte + scale offsets). */
    rocke_value_t** b_byte_off
        = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_sub_n * sizeof(*b_byte_off));
    rocke_value_t** b_scale_off
        = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_sub_n * sizeof(*b_scale_off));
    if(b_byte_off == NULL || b_scale_off == NULL)
    {
        return NULL;
    }
    {
        int sn;
        for(sn = 0; sn < n_sub_n; ++sn)
        {
            rocke_value_t* b_row
                = rocke_b_add(b, wn_base, rocke_b_add(b, rocke_b_const_i32(b, sn * wtn), frag));
            b_byte_off[sn] = rocke_b_mul(b, b_row, rocke_b_const_i32(b, k_packed_stride));
            b_scale_off[sn] = rocke_b_mul(b, b_row, rocke_b_const_i32(b, k_group_stride));
        }
    }

    /* --- acc0 = [zero_vec_f32(8) for _ in range(n_acc)] --- */
    rocke_iter_arg_t* iter_args
        = (rocke_iter_arg_t*)rocke_arena_alloc(&b->arena, (size_t)n_acc * sizeof(*iter_args));
    if(iter_args == NULL)
    {
        return NULL;
    }
    {
        int i;
        for(i = 0; i < n_acc; ++i)
        {
            char* nm = (char*)rocke_arena_alloc(&b->arena, 24);
            if(nm == NULL)
            {
                return NULL;
            }
            snprintf(nm, 24, "acc%d", i);
            iter_args[i].name = nm;
            iter_args[i].init = rocke_b_zero_vec_f32(b, 8);
        }
    }

    rocke_for_t loop
        = rocke_b_scf_for_iter(b, c0, cK, c_tile_k, iter_args, n_acc, "k0", false, true);
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* k0 = loop.iv;
        rocke_value_t* const* accs = loop.iter_vars;

        rocke_value_t* k_grp = rocke_b_div(b, k0, c_group); /* scale group index */
        rocke_value_t* k_half_base = rocke_b_div(b, k0, c2); /* packed-byte base  */

        /* --- step 1: stage the A tile [tile_m x tile_k] into LDS --- */
        {
            int ch;
            for(ch = 0; ch < a_chunks; ++ch)
            {
                /* Python: b.add(b.mul(tid, const(a_chunks*8)), const(ch*8)) --
                 * the mul (incl. its const) is fully evaluated before const(ch*8).
                 * Force that order; C arg evaluation order is unspecified. */
                rocke_value_t* lin_mul = rocke_b_mul(b, tid, rocke_b_const_i32(b, a_chunks * 8));
                rocke_value_t* lin = rocke_b_add(b, lin_mul, rocke_b_const_i32(b, ch * 8));
                rocke_value_t* r = rocke_b_div(b, lin, c_tile_k);
                rocke_value_t* c = rocke_b_mod(b, lin, c_tile_k);
                rocke_value_t* g_idx = rocke_b_add(
                    b, rocke_b_add(b, rocke_b_mul(b, rocke_b_add(b, m0, r), cK), k0), c);
                rocke_value_t* a_vec = rocke_b_global_load_vN_f16(b, A, g_idx, 8, 0);
                rocke_value_t* idxs[2];
                idxs[0] = r;
                idxs[1] = c;
                rocke_b_smem_store_vN_f16(b, a_smem, idxs, 2, a_vec, 8);
            }
        }
        rocke_b_sync(b); /* A tile visible to all waves */

        /* --- step 2: contract the whole group into a group-local f32 accumulator,
         *          feeding UNSCALED int4->f16 fragments to the WMMA atom. --- */
        rocke_value_t** gacc
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_acc * sizeof(*gacc));
        if(gacc == NULL)
        {
            return NULL;
        }
        {
            int i;
            for(i = 0; i < n_acc; ++i)
            {
                gacc[i] = rocke_b_zero_vec_f32(b, 8);
            }
        }

        {
            int ks;
            for(ks = 0; ks < n_ksub; ++ks)
            {
                rocke_value_t* a_col = rocke_b_add(b, rocke_b_const_i32(b, ks * wtk), half_k_elem);
                rocke_value_t** a_frag = (rocke_value_t**)rocke_arena_alloc(
                    &b->arena, (size_t)n_sub_m * sizeof(*a_frag));
                if(a_frag == NULL)
                {
                    return NULL;
                }
                {
                    int sm;
                    for(sm = 0; sm < n_sub_m; ++sm)
                    {
                        rocke_value_t* a_row = rocke_b_add(
                            b, wm_local, rocke_b_add(b, rocke_b_const_i32(b, sm * wtm), frag));
                        rocke_value_t* idxs[2];
                        idxs[0] = a_row;
                        idxs[1] = a_col;
                        a_frag[sm] = rocke_b_smem_load_vN_f16(b, a_smem, idxs, 2, wp.frag_k);
                    }
                }

                int n_bytes = wp.frag_k / 2;
                rocke_value_t* b_byte_k
                    = rocke_b_add(b,
                                  rocke_b_add(b, k_half_base, rocke_b_const_i32(b, ks * (wtk / 2))),
                                  half_k_byte);

                rocke_value_t** b_frag = (rocke_value_t**)rocke_arena_alloc(
                    &b->arena, (size_t)n_sub_n * sizeof(*b_frag));
                if(b_frag == NULL)
                {
                    return NULL;
                }
                {
                    int sn;
                    for(sn = 0; sn < n_sub_n; ++sn)
                    {
                        rocke_value_t* packed = rocke_b_global_load_vN(
                            b, Bp, rocke_b_add(b, b_byte_off[sn], b_byte_k), I8t, n_bytes, 0);
                        rocke_value_t** comps = (rocke_value_t**)rocke_arena_alloc(
                            &b->arena, (size_t)(2 * n_bytes) * sizeof(*comps));
                        if(comps == NULL)
                        {
                            return NULL;
                        }
                        {
                            int j;
                            for(j = 0; j < n_bytes; ++j)
                            {
                                rocke_value_t* byte = rocke_b_vec_extract(b, packed, j);
                                rocke_value_t* lo = NULL;
                                rocke_value_t* hi = NULL;
                                if(rocke_unpack_i4_byte_to_pair_f16(b, byte, &lo, &hi) != ROCKE_OK)
                                {
                                    return NULL;
                                }
                                comps[2 * j] = lo;
                                comps[2 * j + 1] = hi;
                            }
                        }
                        b_frag[sn] = rocke_b_vec_pack(b, comps, 2 * n_bytes, F16t);
                    }
                }

                {
                    rocke_value_t** ng = (rocke_value_t**)rocke_arena_alloc(
                        &b->arena, (size_t)n_acc * sizeof(*ng));
                    if(ng == NULL)
                    {
                        return NULL;
                    }
                    int sm, sn;
                    for(sm = 0; sm < n_sub_m; ++sm)
                    {
                        for(sn = 0; sn < n_sub_n; ++sn)
                        {
                            int idx = sm * n_sub_n + sn;
                            ng[idx]
                                = rocke_opt_wmma(b, wp.wmma_op, a_frag[sm], b_frag[sn], gacc[idx]);
                            if(ng[idx] == NULL)
                            {
                                return NULL;
                            }
                        }
                    }
                    gacc = ng;
                }
            }
        }

        /* --- scale the group accumulator by the per-column group scale and add
         *     into the main accumulator (one vector_fma per col). --- */
        rocke_value_t** scale_vec
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_sub_n * sizeof(*scale_vec));
        if(scale_vec == NULL)
        {
            return NULL;
        }
        {
            int sn;
            for(sn = 0; sn < n_sub_n; ++sn)
            {
                rocke_value_t* s = rocke_b_global_load(
                    b, Sp, rocke_b_add(b, b_scale_off[sn], k_grp), scale_t, 1);
                scale_vec[sn] = rocke_b_vector_splat(b, rocke_b_cast_to_f32(b, s), 8);
            }
        }

        rocke_value_t** nacc
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_acc * sizeof(*nacc));
        if(nacc == NULL)
        {
            return NULL;
        }
        {
            int sm, sn;
            for(sm = 0; sm < n_sub_m; ++sm)
            {
                for(sn = 0; sn < n_sub_n; ++sn)
                {
                    int idx = sm * n_sub_n + sn;
                    nacc[idx] = rocke_b_vector_fma(b, gacc[idx], scale_vec[sn], accs[idx]);
                }
            }
        }

        rocke_b_sync(b); /* all A-tile reads done before next group overwrites a_smem */
        rocke_b_scf_yield(b, nacc, n_acc);
    }
    rocke_b_region_leave(b);

    rocke_value_t* const* results = loop.op->results;

    /* --- step 3: write the column-distributed accumulator to LDS, then store the
     *         whole tile to global as coalesced b128 (8 halves/transaction). --- */
    rocke_value_t* wn_local = rocke_b_mul(b, wave_n, rocke_b_const_i32(b, cols_per_wave));
    {
        int sm, sn, i;
        for(sm = 0; sm < n_sub_m; ++sm)
        {
            rocke_value_t* r0 = rocke_b_add(b, wm_local, rocke_b_const_i32(b, sm * wtm));
            for(sn = 0; sn < n_sub_n; ++sn)
            {
                rocke_value_t* acc = results[sm * n_sub_n + sn];
                rocke_value_t* col = rocke_b_add(
                    b, wn_local, rocke_b_add(b, rocke_b_const_i32(b, sn * wtn), frag));
                for(i = 0; i < 8; ++i)
                {
                    rocke_value_t* h = rocke_b_trunc_f32_to_f16(b, rocke_b_vec_extract(b, acc, i));
                    rocke_value_t* row
                        = rocke_b_add(b, r0, rocke_b_add(b, half_k_elem, rocke_b_const_i32(b, i)));
                    rocke_value_t* idxs[2];
                    idxs[0] = row;
                    idxs[1] = col;
                    rocke_b_smem_store_vN_f16(b, c_smem, idxs, 2, h, 1);
                }
            }
        }
    }
    rocke_b_sync(b);

    rocke_value_t* c_tile_n = rocke_b_const_i32(b, tile_n);
    {
        int ch;
        for(ch = 0; ch < c_chunks; ++ch)
        {
            /* Python: b.add(b.mul(tid, const(c_chunks*8)), const(ch*8)) -- evaluate
             * the mul (incl. its const) before const(ch*8) to match SSA numbering. */
            rocke_value_t* lin_mul = rocke_b_mul(b, tid, rocke_b_const_i32(b, c_chunks * 8));
            rocke_value_t* lin = rocke_b_add(b, lin_mul, rocke_b_const_i32(b, ch * 8));
            rocke_value_t* r = rocke_b_div(b, lin, c_tile_n);
            rocke_value_t* c = rocke_b_mod(b, lin, c_tile_n);
            rocke_value_t* idxs[2];
            idxs[0] = r;
            idxs[1] = c;
            rocke_value_t* v = rocke_b_smem_load_vN_f16(b, c_smem, idxs, 2, 8);
            rocke_value_t* g_idx
                = rocke_b_add(b, rocke_b_add(b, rocke_b_mul(b, rocke_b_add(b, m0, r), cN), n0), c);
            rocke_b_global_store_vN_f16(b, C, g_idx, v, 8, 0);
        }
    }

    return b->kernel;
}

rocke_kernel_def_t* rocke_build_large_n_opt_matmul_nbits_new(rocke_ir_builder_t* b,
                                                             const rocke_matmul_nbits_spec_t* spec,
                                                             const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(rocke_matmul_nbits_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_large_n_opt_matmul_nbits(b, spec, arch);
    });
}
