// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_spatial.cpp -- C99 port of
 * build_direct_depthwise_spatial (rocke/instances/common/conv_direct_grouped.py,
 * lines 2928-3119).
 *
 * Small-group depthwise spatial kernel (groups <= wave_size).
 *
 * Thread layout within each wavefront:
 *   ch        = t_in_wave % groups   -> which channel this thread owns
 *   w_in_wave = t_in_wave // groups  -> W-position offset within the wave
 *
 * Each wave covers n_w_per_wave = wave_size // groups output W positions for
 * ALL channels simultaneously.
 *
 * Block geometry:
 *   threads_per_block = block_waves * wave_size
 *   block_w = block_waves * n_w_per_wave
 *   Grid: (ceil(Wo / block_w), 1, N) — no channel tile.
 *
 * Descriptor notes:
 *   A[N, H, W, total_c] + embed h=(y_iter, strides=(1,)) + embed w=(wo,s_off, strides=(stride,1))
 *   B[total_k, KH, KW, 1]  naive
 *   D[N, Ho, Wo, total_k]  naive
 *
 * Unroll threshold: n_iters * KH * KW <= 20000 (one W-pos per thread).
 * Above the threshold: scf_for_iter over n_groups = ceil(n_iters / KH) groups.
 *
 * Phase functions: rocke_build_direct_depthwise_spatial (full kernel build),
 *                  rocke_build_direct_depthwise_spatial_new (init b + build).
 * Byte-identical to the Python source.
 */
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* Forward declaration for the _new variant's guard helper (C++ only TU). */
#include "rocke/error_boundary.hpp"

/* ===================================================================== *
 *  Internal helpers
 * ===================================================================== */

static int sp_ho(const rocke_direct_conv_problem_t* p)
{
    int s = p->stride > 0 ? p->stride : 1;
    return (p->H + 2 * p->PAD - p->KH) / s + 1;
}

static int sp_wo(const rocke_direct_conv_problem_t* p)
{
    int s = p->stride > 0 ? p->stride : 1;
    return (p->W + 2 * p->PAD - p->KW) / s + 1;
}

/* ===================================================================== *
 *  rocke_build_direct_depthwise_spatial
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_depthwise_spatial(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_spatial_spec_t* spec, const char* arch)
{
    rocke_ir_builder_t* bld = b;
    const rocke_direct_conv_problem_t* p;
    int WAVE, BLOCK_WAVES, THREADS, n_w, BLOCK_W;
    int Ho, Wo, c_stride_dw;
    int n_iters;
    int _use_unroll;
    int total_c, total_k;
    int is_bf16;

    rocke_dconv_params_t params;
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;

    rocke_value_t* c0;
    rocke_value_t* c_wave;
    rocke_value_t* c_Wo;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* zero_f32;

    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* t_in_wave;
    rocke_value_t* ch;
    rocke_value_t* w_in_wave;
    rocke_value_t* bx;
    rocke_value_t* n;
    rocke_value_t* q_out;
    rocke_value_t* w_valid;
    rocke_value_t* q_ok;

    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc;

    const rocke_tensor_descriptor_t* a_desc;
    const rocke_tensor_descriptor_t* b_desc_v;
    const rocke_tensor_descriptor_t* d_desc;

    /* KH x KW preloaded weights (f32), each guarded by w_valid */
    rocke_value_t* weights_f32[ROCKE_DCONV_DW_MAX_KH][ROCKE_DCONV_DW_MAX_KW];

    char reason[ROCKE_ERR_MSG_CAP];

    if(bld == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* Validate */
    if(!rocke_direct_depthwise_spatial_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        if(bld->status == ROCKE_OK)
        {
            bld->status = ROCKE_ERR_VALUE;
        }
        return NULL;
    }

    p = &spec->problem;
    WAVE = spec->wave_size;
    BLOCK_WAVES = spec->block_waves;
    (void)BLOCK_WAVES;
    THREADS = rocke_direct_depthwise_spatial_threads_per_block(spec);
    n_w = rocke_direct_depthwise_spatial_n_w_per_wave(spec);
    BLOCK_W = rocke_direct_depthwise_spatial_block_w(spec);
    c_stride_dw = p->stride > 0 ? p->stride : 1;
    Ho = sp_ho(p);
    Wo = sp_wo(p);
    total_c = rocke_direct_conv_problem_total_c(p);
    /* AOT: the output extents and the channel count live in kernargs now;
     * these stay only for the spec-validation arithmetic above. */
    (void)Wo;
    (void)total_c;
    total_k = rocke_direct_conv_problem_total_k(p);

    n_iters = p->H + p->KH - 1;
    /* AOT: the row count follows the runtime height, so only the
     * grouped-period scf.for form is usable; the build-time-unrolled variant
     * would bake Hi into the trip count. */
    _use_unroll = 0;
    (void)n_iters;
    is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0) ? 1 : 0;

    rocke_attr_set_int(bld, &bld->kernel->attrs, "max_workgroup_size", THREADS);

    /* ---- parameters ---- */
    {
        /* The AOT kernarg block, in conv_abi order. */
        rocke_dconv_emit_params(
            bld, &params, "fwd", rocke_b_io_ir_type(bld, p->dtype ? p->dtype : "fp16"));
        A = params.A;
        Bp = params.Bp;
        D = params.D;
        A_bytes = params.A_bytes;
        B_bytes = params.B_bytes;
        D_bytes = params.D_bytes;
    }

    /* ---- SSA constants ---- */
    c0 = rocke_b_const_i32(bld, 0);
    c_wave = rocke_b_const_i32(bld, WAVE);
    /* AOT: the store guard bounds against the runtime output width. */
    c_Wo = params.p_Wo;
    c_half_bytes = rocke_b_const_i32(bld, 2);
    oob_sentinel = rocke_b_const_i32(bld, ((int64_t)1 << 31) - 1);
    zero_f32 = rocke_b_const_f32(bld, 0.0f);

    /* ---- thread / wave / lane decode ---- */
    tid = rocke_b_thread_id_x(bld);
    wave_id = rocke_b_div(bld, tid, c_wave);
    t_in_wave = rocke_b_mod(bld, tid, c_wave);

    ch = rocke_b_mod(bld, t_in_wave, params.p_groups);
    w_in_wave = rocke_b_div(bld, t_in_wave, params.p_groups);

    /* Grid: bx=W-tile, bz=batch */
    bx = rocke_b_block_id_x(bld);
    n = rocke_b_block_id_z(bld);

    /* q_out = bx*BLOCK_W + wave_id*n_w + w_in_wave
     * Force Python left-to-right: const(BLOCK_W), mul, const(n_w), mul, add, add. */
    {
        rocke_value_t* c_bw = rocke_b_const_i32(bld, BLOCK_W);
        rocke_value_t* mul_bx = rocke_b_mul(bld, bx, c_bw);
        rocke_value_t* c_nw = rocke_b_const_i32(bld, n_w);
        rocke_value_t* mul_wv = rocke_b_mul(bld, wave_id, c_nw);
        rocke_value_t* inner = rocke_b_add(bld, mul_wv, w_in_wave);
        q_out = rocke_b_add(bld, mul_bx, inner);
    }

    /* Guard: wasted threads when groups * n_w < wave_size */
    w_valid = rocke_b_cmp_lt(bld, w_in_wave, rocke_b_const_i32(bld, n_w));
    q_ok = rocke_b_land(bld, rocke_b_cmp_lt(bld, q_out, c_Wo), w_valid);

    /* ---- buffer resources ---- */
    a_rsrc = rocke_b_buffer_rsrc(bld, A, A_bytes);
    b_rsrc = rocke_b_buffer_rsrc(bld, Bp, B_bytes);
    d_rsrc = rocke_b_buffer_rsrc(bld, D, D_bytes);

    /* ---- descriptors ---- */
    {
        /* a_desc with runtime extents; PAD and stride stay build-time. */
        rocke_dynamic_tensor_descriptor_t* a_dyn
            = rocke_dconv_a_descriptor_dynamic(bld, &params, p->PAD, c_stride_dw, "wo", "s_off");
        if(!a_dyn)
            return NULL;
        a_desc = &a_dyn->base;
    }

    {
        /* b_desc_v = naive("B", [total_k, KH, KW, 1]) */
        static const char* const b_coords[4] = {"k", "r", "s", "c"};
        int b_lengths[4];
        b_lengths[0] = total_k;
        b_lengths[1] = p->KH;
        b_lengths[2] = p->KW;
        b_lengths[3] = 1;
        b_desc_v = rocke_tensor_descriptor_naive(bld, "B", b_lengths, 4, NULL, b_coords, 4);
    }

    {
        /* d_desc = D[N, Ho, Wo, total_k] with runtime extents. */
        rocke_dynamic_tensor_descriptor_t* d_dyn = rocke_dconv_d_descriptor_dynamic(bld, &params);
        if(!d_dyn)
            return NULL;
        d_desc = &d_dyn->base;
    }

    /* ---- Preload weights: KH * KW f32 per thread, guarded by w_valid ---- */
    {
        int r_const, s_const;
        for(r_const = 0; r_const < p->KH; ++r_const)
        {
            for(s_const = 0; s_const < p->KW; ++s_const)
            {
                rocke_value_t* w_off = NULL;
                rocke_value_t* w_valid_desc;
                rocke_value_t* safe_w;
                rocke_value_t* w_h;
                const char* bn[4] = {"k", "r", "s", "c"};
                rocke_value_t* bv[4];

                bv[0] = ch;
                bv[1] = rocke_b_const_i32(bld, r_const);
                bv[2] = rocke_b_const_i32(bld, s_const);
                bv[3] = c0;
                rocke_transforms_descriptor_offset(bld, b_desc_v, bn, bv, 4, &w_off, &w_valid_desc);

                safe_w = rocke_b_select(
                    bld, w_valid, rocke_b_mul(bld, w_off, c_half_bytes), oob_sentinel);
                w_h = is_bf16 ? rocke_b_buffer_load_bf16(bld, b_rsrc, safe_w, c0)
                              : rocke_b_buffer_load_f16(bld, b_rsrc, safe_w, c0);
                weights_f32[r_const][s_const]
                    = rocke_b_select(bld, w_valid, rocke_b_cast_to_f32(bld, w_h), zero_f32);
            }
        }
    }

    /* ================================================================== *
     *  H-streaming loop
     * ================================================================== */
    if(_use_unroll)
    {
        /* ---- static unroll path ---- */
        /* acc[slot]: KH f32 circular accumulators (one W-pos per thread) */
        rocke_value_t* acc[ROCKE_DCONV_DW_MAX_KH];
        int y, slot, r_const, s_const;

        for(slot = 0; slot < p->KH; ++slot)
            acc[slot] = zero_f32;

        for(y = 0; y < n_iters; ++y)
        {
            rocke_value_t* y_i = rocke_b_const_i32(bld, y);
            int p_flush_val = y - (p->KH - 1);
            int P_FLUSH = ((p_flush_val % p->KH) + p->KH) % p->KH;

            /* FMA: for s_const: load A; for r_const: fma */
            for(s_const = 0; s_const < p->KW; ++s_const)
            {
                rocke_value_t* a_off = NULL;
                rocke_value_t* valid = NULL;
                rocke_value_t* ok;
                rocke_value_t* safe_off;
                rocke_value_t* a_h;
                rocke_value_t* a_f32;
                const char* on[5];
                rocke_value_t* ov[5];

                on[0] = "n";
                ov[0] = n;
                on[1] = "y_iter";
                ov[1] = y_i;
                on[2] = "wo";
                ov[2] = q_out;
                on[3] = "s_off";
                ov[3] = rocke_b_const_i32(bld, s_const);
                on[4] = "c";
                ov[4] = ch;
                rocke_transforms_descriptor_offset(bld, a_desc, on, ov, 5, &a_off, &valid);

                ok = rocke_b_land(bld, valid, q_ok);
                safe_off
                    = rocke_b_select(bld, ok, rocke_b_mul(bld, a_off, c_half_bytes), oob_sentinel);
                a_h = is_bf16 ? rocke_b_buffer_load_bf16(bld, a_rsrc, safe_off, c0)
                              : rocke_b_buffer_load_f16(bld, a_rsrc, safe_off, c0);
                a_f32 = rocke_b_select(bld, ok, rocke_b_cast_to_f32(bld, a_h), zero_f32);

                for(r_const = 0; r_const < p->KH; ++r_const)
                {
                    int p_idx = (((y - r_const) % p->KH) + p->KH) % p->KH;
                    acc[p_idx] = rocke_b_fma(bld, weights_f32[r_const][s_const], a_f32, acc[p_idx]);
                }
            }

            /* Flush: stride-aligned, in [0,H) and ho_row < Ho */
            if(0 <= p_flush_val && p_flush_val < p->H && p_flush_val % c_stride_dw == 0)
            {
                int ho_row = p_flush_val / c_stride_dw;
                if(ho_row < Ho)
                {
                    rocke_value_t* d_off = NULL;
                    rocke_value_t* d_valid = NULL;
                    rocke_value_t* safe_d;
                    const char* dn[4];
                    rocke_value_t* dv[4];

                    dn[0] = "n";
                    dv[0] = n;
                    dn[1] = "h";
                    dv[1] = rocke_b_const_i32(bld, ho_row);
                    dn[2] = "w";
                    dv[2] = q_out;
                    dn[3] = "k";
                    dv[3] = ch;
                    rocke_transforms_descriptor_offset(bld, d_desc, dn, dv, 4, &d_off, &d_valid);

                    safe_d = rocke_b_select(
                        bld, q_ok, rocke_b_mul(bld, d_off, c_half_bytes), oob_sentinel);
                    if(is_bf16)
                        rocke_b_buffer_store_bf16(
                            bld, d_rsrc, safe_d, c0, rocke_b_trunc_f32_to_bf16(bld, acc[P_FLUSH]));
                    else
                        rocke_b_buffer_store_f16(
                            bld, d_rsrc, safe_d, c0, rocke_b_trunc_f32_to_f16(bld, acc[P_FLUSH]));
                }
            }

            /* Unconditional reset */
            acc[P_FLUSH] = zero_f32;
        }
    }
    else
    {
        /* ---- scf_for_iter group loop path ---- */
        int KH = p->KH;
        int num_iargs = KH;
        rocke_iter_arg_t* iargs;
        rocke_for_t group_loop;
        rocke_value_t** new_accs;
        rocke_value_t* c1;
        rocke_value_t* c_KH;
        rocke_value_t* c_stride_rv;
        rocke_value_t* n_iters_v;
        rocke_value_t* n_groups_v;
        int j;

        {
            char(*name_store)[32] = (char(*)[32])alloca((size_t)num_iargs * 32 * sizeof(char));
            int kh;

            iargs = (rocke_iter_arg_t*)alloca((size_t)num_iargs * sizeof(rocke_iter_arg_t));
            new_accs = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));

            for(kh = 0; kh < KH; ++kh)
            {
                snprintf(name_store[kh], 32, "sp_acc_%d", kh);
                iargs[kh].name = name_store[kh];
                iargs[kh].init = zero_f32;
            }
        }

        c1 = rocke_b_const_i32(bld, 1);
        c_KH = rocke_b_const_i32(bld, KH);
        c_stride_rv = rocke_b_const_i32(bld, c_stride_dw);
        /* n_iters = Hi + KH - 1 rows; the loop walks them KH at a time. */
        n_iters_v = rocke_b_add(bld, params.p_Hi, rocke_b_const_i32(bld, KH - 1));
        n_groups_v
            = rocke_b_div(bld, rocke_b_add(bld, n_iters_v, rocke_b_const_i32(bld, KH - 1)), c_KH);

        group_loop = rocke_b_scf_for_iter(bld,
                                          c0,
                                          n_groups_v,
                                          c1,
                                          iargs,
                                          num_iargs,
                                          "sp_grp",
                                          /*unroll=*/false,
                                          /*elide_trailing_barrier=*/false);
        rocke_b_region_enter(bld, group_loop.body);

        {
            int i;
            for(i = 0; i < num_iargs; ++i)
                new_accs[i] = group_loop.iter_vars[i];
        }

        for(j = 0; j < KH; ++j)
        {
            rocke_value_t* y_j;
            rocke_value_t* j_valid;
            int s_const, r_const;
            int P_FLUSH_j = (j + 1) % KH;
            rocke_value_t* p_flush_rv;
            rocke_value_t* flush_ge;
            rocke_value_t* should_flush;
            rocke_value_t* ho_row_j;
            rocke_value_t* store_ok;
            rocke_value_t* acc_val;
            rocke_value_t* d_off;
            rocke_value_t* d_valid;
            rocke_value_t* safe_d;
            const char* dn[4];
            rocke_value_t* dv[4];

            /* y_j = grp_iv*c_KH + j -- emit the mul first, then const(j), to
             * match Python's left-to-right evaluation (C++ arg order is
             * unspecified and GCC evaluates right-to-left here). */
            {
                rocke_value_t* mul_gk = rocke_b_mul(bld, group_loop.iv, c_KH);
                rocke_value_t* cj = rocke_b_const_i32(bld, j);
                y_j = rocke_b_add(bld, mul_gk, cj);
            }
            j_valid = rocke_b_cmp_lt(bld, y_j, n_iters_v);

            for(s_const = 0; s_const < p->KW; ++s_const)
            {
                rocke_value_t* a_off = NULL;
                rocke_value_t* valid = NULL;
                rocke_value_t* ok;
                rocke_value_t* safe_off;
                rocke_value_t* a_h;
                rocke_value_t* a_f32;
                const char* on[5];
                rocke_value_t* ov[5];

                on[0] = "n";
                ov[0] = n;
                on[1] = "y_iter";
                ov[1] = y_j;
                on[2] = "wo";
                ov[2] = q_out;
                on[3] = "s_off";
                ov[3] = rocke_b_const_i32(bld, s_const);
                on[4] = "c";
                ov[4] = ch;
                rocke_transforms_descriptor_offset(bld, a_desc, on, ov, 5, &a_off, &valid);

                ok = rocke_b_land(bld, rocke_b_land(bld, valid, j_valid), q_ok);
                safe_off
                    = rocke_b_select(bld, ok, rocke_b_mul(bld, a_off, c_half_bytes), oob_sentinel);
                a_h = is_bf16 ? rocke_b_buffer_load_bf16(bld, a_rsrc, safe_off, c0)
                              : rocke_b_buffer_load_f16(bld, a_rsrc, safe_off, c0);
                a_f32 = rocke_b_select(bld, ok, rocke_b_cast_to_f32(bld, a_h), zero_f32);

                for(r_const = 0; r_const < KH; ++r_const)
                {
                    int p_idx = ((j - r_const + KH) % KH); /* STATIC */
                    new_accs[p_idx]
                        = rocke_b_fma(bld, weights_f32[r_const][s_const], a_f32, new_accs[p_idx]);
                }
            }

            /* Flush / store */
            p_flush_rv = rocke_b_add(bld, y_j, rocke_b_const_i32(bld, -(KH - 1)));

            if(j >= KH - 1)
            {
                flush_ge = j_valid;
            }
            else
            {
                flush_ge = rocke_b_land(bld, rocke_b_cmp_lt(bld, c0, group_loop.iv), j_valid);
            }

            if(c_stride_dw == 1)
            {
                should_flush = flush_ge;
            }
            else
            {
                rocke_value_t* flush_stride
                    = rocke_b_cmp_eq(bld, rocke_b_mod(bld, p_flush_rv, c_stride_rv), c0);
                should_flush = rocke_b_land(bld, flush_ge, flush_stride);
            }

            ho_row_j = rocke_b_div(bld, p_flush_rv, c_stride_rv);
            store_ok = rocke_b_land(bld, q_ok, should_flush);
            acc_val = new_accs[P_FLUSH_j];

            d_off = NULL;
            d_valid = NULL;
            dn[0] = "n";
            dv[0] = n;
            dn[1] = "h";
            dv[1] = ho_row_j;
            dn[2] = "w";
            dv[2] = q_out;
            dn[3] = "k";
            dv[3] = ch;
            rocke_transforms_descriptor_offset(bld, d_desc, dn, dv, 4, &d_off, &d_valid);

            safe_d = rocke_b_select(
                bld, store_ok, rocke_b_mul(bld, d_off, c_half_bytes), oob_sentinel);
            if(is_bf16)
                rocke_b_buffer_store_bf16(
                    bld, d_rsrc, safe_d, c0, rocke_b_trunc_f32_to_bf16(bld, acc_val));
            else
                rocke_b_buffer_store_f16(
                    bld, d_rsrc, safe_d, c0, rocke_b_trunc_f32_to_f16(bld, acc_val));

            /* Unconditional static reset */
            new_accs[P_FLUSH_j] = zero_f32;
        }

        rocke_b_scf_yield(bld, new_accs, num_iargs);
        rocke_b_region_leave(bld);
    }

    return rocke_ir_builder_kernel(bld);
}

/* ===================================================================== *
 *  rocke_build_direct_depthwise_spatial_new
 *
 *  Initialises `b` with spec.kernel_name(), then calls the main build.
 *  Caller owns `b` and must free with rocke_ir_builder_free().
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_depthwise_spatial_new(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_spatial_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_direct_depthwise_spatial_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_direct_depthwise_spatial(b, spec, arch);
    });
}
