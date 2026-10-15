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
 * The H rows stream through scf_for_iter over KH-row periods; the trip count
 * follows the runtime height (AOT), so there is no build-time-unrolled form.
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

static int sp_wo(const rocke_direct_conv_problem_t* p)
{
    int s = p->stride > 0 ? p->stride : 1;
    return (p->W + 2 * p->PAD - p->KW) / s + 1;
}

/* Raw input values of row y, one per filter column (Python: load_row).
 * The per-lane column offsets come precomputed (OOB sentinel where the tap
 * falls outside the image or the lane is idle); the wave-uniform row offset
 * travels in the load's scalar offset. Halo rows load from the sentinel with
 * a zero scalar offset, which the buffer descriptor returns as zero. */
static void sp_load_row(rocke_ir_builder_t* bld,
                        rocke_value_t* y,
                        int PAD,
                        int KW,
                        const rocke_dconv_params_t* params,
                        rocke_value_t* n,
                        rocke_value_t* const* col_offs,
                        rocke_value_t* a_rsrc,
                        rocke_value_t* c_half_bytes,
                        rocke_value_t* oob_sentinel,
                        rocke_value_t* c0,
                        int is_bf16,
                        rocke_value_t** out)
{
    rocke_value_t* hi;
    rocke_value_t* row_ok;
    rocke_value_t* row_off;
    int s_const;

    {
        rocke_value_t* c_pad = rocke_b_const_i32(bld, -PAD);
        hi = rocke_b_add(bld, y, c_pad);
    }
    {
        rocke_value_t* ge = rocke_b_cmp_ge(bld, hi, c0);
        rocke_value_t* lt = rocke_b_cmp_lt(bld, hi, params->p_Hi);
        row_ok = rocke_b_land(bld, ge, lt);
    }
    {
        rocke_value_t* off_n = rocke_b_mul(bld, n, params->p_A_stride_n);
        rocke_value_t* off_h = rocke_b_mul(bld, hi, params->p_A_stride_hi);
        row_off = rocke_b_mul(bld, rocke_b_add(bld, off_n, off_h), c_half_bytes);
    }
    row_off = rocke_b_select(bld, row_ok, row_off, c0);

    for(s_const = 0; s_const < KW; ++s_const)
    {
        rocke_value_t* safe_off = rocke_b_select(bld, row_ok, col_offs[s_const], oob_sentinel);
        out[s_const] = is_bf16 ? rocke_b_buffer_load_bf16(bld, a_rsrc, safe_off, row_off)
                               : rocke_b_buffer_load_f16(bld, a_rsrc, safe_off, row_off);
    }
}

/* ===================================================================== *
 *  rocke_build_direct_depthwise_spatial
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_depthwise_spatial(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_spatial_spec_t* spec, const char* arch)
{
    rocke_ir_builder_t* bld = b;
    const rocke_direct_conv_problem_t* p;
    int WAVE, BLOCK_WAVES, THREADS;
    int Wo, c_stride_dw;
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
    rocke_value_t* n_w;
    rocke_value_t* q_out;
    rocke_value_t* w_valid;
    rocke_value_t* q_ok;

    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc;

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
    /* The weight table is sized for filters up to the depthwise limit. */
    if(p->KH > ROCKE_DCONV_DW_MAX_KH || p->KW > ROCKE_DCONV_DW_MAX_KW)
    {
        if(bld->status == ROCKE_OK)
        {
            bld->status = ROCKE_ERR_VALUE;
        }
        return NULL;
    }
    WAVE = spec->wave_size;
    BLOCK_WAVES = spec->block_waves;
    THREADS = rocke_direct_depthwise_spatial_threads_per_block(spec);
    c_stride_dw = p->stride > 0 ? p->stride : 1;
    Wo = sp_wo(p);
    total_c = rocke_direct_conv_problem_total_c(p);
    /* AOT: the output extents and the channel count live in kernargs now;
     * these stay only for the spec-validation arithmetic above. */
    (void)Wo;
    (void)total_c;
    total_k = rocke_direct_conv_problem_total_k(p);

    is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0) ? 1 : 0;

    rocke_attr_set_int(bld, &bld->kernel->attrs, "max_workgroup_size", THREADS);
    /* Two waves per SIMD: the KH x KW weights and the prefetch window push
     * larger filters just over the one-wave threshold; two waves are faster. */
    rocke_attr_set_int(bld, &bld->kernel->attrs, "waves_per_eu", 2);

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

    /* AOT: groups is a kernarg, so the W positions per wave (and with them the
     * block's W tile) are derived at run time; the host sizes the grid with
     * the same n_w = wave_size / groups.
     * q_out = bx*(BLOCK_WAVES*n_w) + wave_id*n_w + w_in_wave
     * Force Python left-to-right: const(BLOCK_WAVES), mul, mul, mul, add, add. */
    n_w = rocke_b_div(bld, c_wave, params.p_groups);
    {
        rocke_value_t* c_bwv = rocke_b_const_i32(bld, BLOCK_WAVES);
        rocke_value_t* block_w = rocke_b_mul(bld, c_bwv, n_w);
        rocke_value_t* mul_bx = rocke_b_mul(bld, bx, block_w);
        rocke_value_t* mul_wv = rocke_b_mul(bld, wave_id, n_w);
        rocke_value_t* inner = rocke_b_add(bld, mul_wv, w_in_wave);
        q_out = rocke_b_add(bld, mul_bx, inner);
    }

    /* Guard: wasted threads when groups * n_w < wave_size */
    w_valid = rocke_b_cmp_lt(bld, w_in_wave, n_w);
    q_ok = rocke_b_land(bld, rocke_b_cmp_lt(bld, q_out, c_Wo), w_valid);

    /* ---- buffer resources ---- */
    a_rsrc = rocke_b_buffer_rsrc(bld, A, A_bytes);
    b_rsrc = rocke_b_buffer_rsrc(bld, Bp, B_bytes);
    d_rsrc = rocke_b_buffer_rsrc(bld, D, D_bytes);

    /* ---- descriptors ---- */
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
    /* AOT: the row count follows the runtime height, so the rows stream
     * through an scf.for over KH-row periods; a build-time unroll would
     * bake Hi into the trip count. The input rows are software-pipelined PF
     * rows ahead (see the Python source for why). */
    {
        /* ---- scf_for_iter group loop path ---- */
        int KH = p->KH;
        int KW = p->KW;
        int PF = (16 + KW - 1) / KW < KH ? (16 + KW - 1) / KW : KH;
        int num_iargs = KH + PF * KW;
        rocke_iter_arg_t* iargs;
        rocke_for_t group_loop;
        rocke_value_t** new_accs;
        /* window[k * KW + s]: rows in flight, oldest first; one spare row
         * slot for the row a step appends before it drops the oldest. */
        rocke_value_t** window;
        rocke_value_t* a_row[ROCKE_DCONV_DW_MAX_KW];
        rocke_value_t* col_offs[ROCKE_DCONV_DW_MAX_KW];
        rocke_value_t* c1;
        rocke_value_t* c_KH;
        rocke_value_t* c_stride_rv;
        rocke_value_t* n_iters_v;
        rocke_value_t* n_groups_v;
        int j;

        window = (rocke_value_t**)alloca((size_t)(PF + 1) * KW * sizeof(rocke_value_t*));

        c1 = rocke_b_const_i32(bld, 1);
        c_KH = rocke_b_const_i32(bld, KH);
        c_stride_rv = rocke_b_const_i32(bld, c_stride_dw);
        /* n_iters = Hi + KH - 1 rows; the loop walks them KH at a time. */
        n_iters_v = rocke_b_add(bld, params.p_Hi, rocke_b_const_i32(bld, KH - 1));
        n_groups_v
            = rocke_b_div(bld, rocke_b_add(bld, n_iters_v, rocke_b_const_i32(bld, KH - 1)), c_KH);

        /* Per-lane column offsets, hoisted out of the loop (Python: col_offs). */
        {
            int s_const;
            for(s_const = 0; s_const < KW; ++s_const)
            {
                rocke_value_t* wi;
                rocke_value_t* col_ok;
                rocke_value_t* col_off;
                {
                    rocke_value_t* q_s = rocke_b_mul(bld, q_out, c_stride_rv);
                    rocke_value_t* c_s = rocke_b_const_i32(bld, s_const - p->PAD);
                    wi = rocke_b_add(bld, q_s, c_s);
                }
                {
                    rocke_value_t* ge = rocke_b_cmp_ge(bld, wi, c0);
                    rocke_value_t* lt = rocke_b_cmp_lt(bld, wi, params.p_Wi);
                    col_ok = rocke_b_land(bld, rocke_b_land(bld, ge, lt), q_ok);
                }
                {
                    rocke_value_t* w_off = rocke_b_mul(bld, wi, params.p_A_stride_wi);
                    col_off = rocke_b_mul(bld, rocke_b_add(bld, w_off, ch), c_half_bytes);
                }
                col_offs[s_const] = rocke_b_select(bld, col_ok, col_off, oob_sentinel);
            }
        }

        {
            char(*name_store)[32] = (char(*)[32])alloca((size_t)num_iargs * 32 * sizeof(char));
            int kh, k, s_const, idx;

            iargs = (rocke_iter_arg_t*)alloca((size_t)num_iargs * sizeof(rocke_iter_arg_t));
            new_accs = (rocke_value_t**)alloca((size_t)KH * sizeof(rocke_value_t*));

            for(kh = 0; kh < KH; ++kh)
            {
                snprintf(name_store[kh], 32, "sp_acc_%d", kh);
                iargs[kh].name = name_store[kh];
                iargs[kh].init = zero_f32;
            }
            /* Rows 0 .. PF-1: in flight when the loop starts. */
            idx = KH;
            for(k = 0; k < PF; ++k)
            {
                sp_load_row(bld,
                            rocke_b_const_i32(bld, k),
                            p->PAD,
                            KW,
                            &params,
                            n,
                            col_offs,
                            a_rsrc,
                            c_half_bytes,
                            oob_sentinel,
                            c0,
                            is_bf16,
                            window + k * KW);
                for(s_const = 0; s_const < KW; ++s_const)
                {
                    snprintf(name_store[idx], 32, "sp_a_r%d_s%d", k, s_const);
                    iargs[idx].name = name_store[idx];
                    iargs[idx].init = window[k * KW + s_const];
                    ++idx;
                }
            }
        }

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
            for(i = 0; i < KH; ++i)
                new_accs[i] = group_loop.iter_vars[i];
            for(i = 0; i < PF * KW; ++i)
                window[i] = group_loop.iter_vars[KH + i];
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

            /* Issue row y_j + PF, then compute row y_j (the oldest in flight). */
            {
                rocke_value_t* cpf = rocke_b_const_i32(bld, PF);
                sp_load_row(bld,
                            rocke_b_add(bld, y_j, cpf),
                            p->PAD,
                            KW,
                            &params,
                            n,
                            col_offs,
                            a_rsrc,
                            c_half_bytes,
                            oob_sentinel,
                            c0,
                            is_bf16,
                            window + PF * KW);
            }
            for(s_const = 0; s_const < KW; ++s_const)
                a_row[s_const] = rocke_b_cast_to_f32(bld, window[s_const]);
            memmove(window, window + KW, (size_t)PF * KW * sizeof(rocke_value_t*));

            for(s_const = 0; s_const < KW; ++s_const)
            {
                rocke_value_t* a_f32 = a_row[s_const];

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

        /* An empty asm on each accumulator keeps LLVM's SLP vectorizer from
         * pairing the loop-carried FMA chains into v_pk_fma_f32 (see Python). */
        {
            const rocke_type_t* f32_ty = rocke_f32();
            rocke_inline_asm_opts_t opts;
            int i;
            memset(&opts, 0, sizeof opts);
            opts.sideeffect = false;
            opts.sideeffect_set = true;
            for(i = 0; i < KH; ++i)
            {
                rocke_op_t* asm_op
                    = rocke_b_inline_asm(bld, "", "=v,0", &new_accs[i], 1, &f32_ty, 1, &opts);
                new_accs[i] = asm_op ? asm_op->results[0] : NULL;
            }
        }
        {
            rocke_value_t** yields
                = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));
            int i;
            for(i = 0; i < KH; ++i)
                yields[i] = new_accs[i];
            for(i = 0; i < PF * KW; ++i)
                yields[KH + i] = window[i];
            rocke_b_scf_yield(bld, yields, num_iargs);
        }
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
