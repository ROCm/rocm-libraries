// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_dgrad.cpp -- C99 port of build_direct_conv_dgrad
 * and build_direct_depthwise_dgrad (rocke/instances/common/conv_direct_grouped.py).
 *
 * Both kernels use scalar FMA only — no MFMA, no LDS.
 *
 * IMPORTANT: every call that has side effects on the IR builder (rocke_b_*) must
 * be issued as a separate statement so that the C compiler cannot reorder them.
 * The Python evaluates sub-expressions left-to-right, so the C port must mirror
 * that order explicitly (unspecified C argument-evaluation order would otherwise
 * produce different IR numbering → different LLVM text → parity failure).
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  HELPERS — tap validity (stride-aware boundary check)
 *
 *  All sub-expressions are issued as sequential statements to preserve the
 *  Python left-to-right emission order (ge first, then const(Ho/Wo), then
 *  lt, then land).
 *
 *  c_st — pre-emitted const_i32(stride) value from the caller, or NULL
 *          when stride==1.  Python emits const_i32(stride) ONCE before the
 *          H/W loop for stride>1; passing it in here avoids re-emitting it
 *          on every tap and keeps SSA numbering identical to Python.
 * ===================================================================== */

/* Stride=1 path  (Python):
 *   valid = land(cmp_ge(hi_p_r, c0),            <- ge first
 *               cmp_lt(hi_p_r, const_i32(Ho)))  <- const(Ho) then lt
 *
 * Stride>1 path  (Python):
 *   ho    = div(hi_p_r, c_st)                   <- div first (c_st pre-emitted)
 *   valid = land(
 *               cmp_ge(hi_p_r, c0),              <- ge
 *               land(
 *                   cmp_eq(mod(hi_p_r, c_st), c0),       <- mod, eq
 *                   cmp_lt(ho, const_i32(Ho)),            <- const(Ho), lt
 *               ),                               <- inner land
 *           )                                    <- outer land
 */
static void tap_valid_h(rocke_ir_builder_t* b,
                        rocke_value_t* hi_p_r,
                        rocke_value_t* c0,
                        rocke_value_t* c_st, /* NULL for stride==1 */
                        int stride,
                        int Ho,
                        rocke_value_t** out_ho,
                        rocke_value_t** out_valid)
{
    if(stride == 1)
    {
        /* Python: land(cmp_ge(hi_p_r, c0), cmp_lt(hi_p_r, const(Ho)))
         * ge is left arg → emitted first; const(Ho) is inside the right arg. */
        rocke_value_t* ge = rocke_b_cmp_ge(b, hi_p_r, c0);
        rocke_value_t* c_Ho = rocke_b_const_i32(b, Ho);
        rocke_value_t* lt = rocke_b_cmp_lt(b, hi_p_r, c_Ho);
        *out_ho = hi_p_r;
        *out_valid = rocke_b_land(b, ge, lt);
    }
    else
    {
        /* c_st was emitted by the caller before the H-loop (Python line 3630). */
        rocke_value_t* ho = rocke_b_div(b, hi_p_r, c_st);
        rocke_value_t* ge = rocke_b_cmp_ge(b, hi_p_r, c0);
        /* Python inner land args: mod/eq are left, const(Ho)/lt are right */
        rocke_value_t* mod_v = rocke_b_mod(b, hi_p_r, c_st);
        rocke_value_t* eq = rocke_b_cmp_eq(b, mod_v, c0);
        rocke_value_t* c_Ho = rocke_b_const_i32(b, Ho);
        rocke_value_t* lt = rocke_b_cmp_lt(b, ho, c_Ho);
        rocke_value_t* inner = rocke_b_land(b, eq, lt);
        *out_ho = ho;
        *out_valid = rocke_b_land(b, ge, inner);
    }
}

/* Same for the W dimension. */
static void tap_valid_w(rocke_ir_builder_t* b,
                        rocke_value_t* wi_p_s,
                        rocke_value_t* c0,
                        rocke_value_t* c_st, /* NULL for stride==1 */
                        int stride,
                        int Wo,
                        rocke_value_t** out_wo,
                        rocke_value_t** out_valid)
{
    if(stride == 1)
    {
        rocke_value_t* ge = rocke_b_cmp_ge(b, wi_p_s, c0);
        rocke_value_t* c_Wo = rocke_b_const_i32(b, Wo);
        rocke_value_t* lt = rocke_b_cmp_lt(b, wi_p_s, c_Wo);
        *out_wo = wi_p_s;
        *out_valid = rocke_b_land(b, ge, lt);
    }
    else
    {
        rocke_value_t* wo = rocke_b_div(b, wi_p_s, c_st);
        rocke_value_t* ge = rocke_b_cmp_ge(b, wi_p_s, c0);
        rocke_value_t* mod_v = rocke_b_mod(b, wi_p_s, c_st);
        rocke_value_t* eq = rocke_b_cmp_eq(b, mod_v, c0);
        rocke_value_t* c_Wo = rocke_b_const_i32(b, Wo);
        rocke_value_t* lt = rocke_b_cmp_lt(b, wo, c_Wo);
        rocke_value_t* inner = rocke_b_land(b, eq, lt);
        *out_wo = wo;
        *out_valid = rocke_b_land(b, ge, inner);
    }
}

/* ===================================================================== *
 *  build_direct_conv_dgrad
 *
 *  Ports Python lines 3558-3747 (build_direct_conv_dgrad).
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_conv_dgrad(rocke_ir_builder_t* b,
                                                  const rocke_direct_conv_dgrad_spec_t* spec,
                                                  const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    int r_const, s_const, j;

    if(rocke_direct_conv_dgrad_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }
    if(!rocke_direct_conv_dgrad_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }

    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int BLOCK_W = spec->block_q;
    const int BLOCK_WAVES = spec->block_groups;
    const int WAVE = spec->wave_size;
    const int THREADS = BLOCK_WAVES * WAVE;
    const int BLOCK_CH = BLOCK_WAVES * WAVE;
    const int Ho = (p->H + 2 * p->PAD - p->KH) / p->stride + 1;
    const int Wo = (p->W + 2 * p->PAD - p->KW) / p->stride + 1;
    const int total_c = p->groups * p->cpg;
    const int total_k = p->groups * p->kpg;
    const int c_stride = p->stride;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);

    /* Params — io_type = _io_type(p.dtype): f16 or bf16 IR type. */
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    const rocke_type_t* ioptr = rocke_ptr_type(b, io_type, "global");
    const int is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0);
    rocke_param_opts_t ro = {0};
    ro.noalias = true;
    ro.noalias_set = true;
    ro.readonly = true;
    ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;
    rocke_param_opts_t wo_opts = {0};
    wo_opts.noalias = true;
    wo_opts.noalias_set = true;
    wo_opts.writeonly = true;
    wo_opts.writeonly_set = true;
    wo_opts.align = 16;
    wo_opts.align_set = true;
    rocke_param_opts_t none = {0};

    rocke_value_t* A = rocke_b_param(b, "A", ioptr, &ro);
    rocke_value_t* Bp = rocke_b_param(b, "B", ioptr, &ro);
    rocke_value_t* D = rocke_b_param(b, "D", ioptr, &wo_opts);
    rocke_value_t* A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
    rocke_value_t* B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
    rocke_value_t* D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);

    /* Constants — emitted in Python source order (lines 3579-3586):
     * c0, c1, c_total_c, c_total_k, c_half_bytes, c_wave, oob_sentinel, zero_f32 */
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_total_c = rocke_b_const_i32(b, total_c);
    rocke_value_t* c_total_k = rocke_b_const_i32(b, total_k); /* emitted, unused in body */
    rocke_value_t* c_half_bytes = rocke_b_const_i32(b, 2);
    rocke_value_t* c_wave = rocke_b_const_i32(b, WAVE);
    rocke_value_t* oob_sentinel = rocke_b_const_i32(b, (int32_t)(((int64_t)1 << 31) - 1));
    rocke_value_t* zero_f32 = rocke_b_const_f32(b, 0.0f);

    /* Thread / wave / lane */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* wave_id = rocke_b_div(b, tid, c_wave);
    rocke_value_t* lane = rocke_b_mod(b, tid, c_wave);

    /* Grid: bx=Wi-tile, by=c_in-tile, bz=n.
     * Each workgroup loops over all H rows via the scf_for below (grid.z = N). */
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* bz = rocke_b_block_id_z(b);
    rocke_value_t* n = bz;

    rocke_value_t* wi_tile_start = rocke_b_mul(b, bx, rocke_b_const_i32(b, BLOCK_W));

    /* c_in = by * BLOCK_CH + wave_id * WAVE + lane
     * Python order: outer (by * BLOCK_CH) evaluated before inner */
    rocke_value_t* c_in_outer = rocke_b_mul(b, by, rocke_b_const_i32(b, BLOCK_CH));
    rocke_value_t* c_in_mul = rocke_b_mul(b, wave_id, c_wave);
    rocke_value_t* c_in_inner = rocke_b_add(b, c_in_mul, lane);
    rocke_value_t* c_in = rocke_b_add(b, c_in_outer, c_in_inner);
    rocke_value_t* c_in_ok = rocke_b_cmp_lt(b, c_in, c_total_c);

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, A, A_bytes);
    rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, Bp, B_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, D, D_bytes);

    /* dY descriptor: A[N, Ho, Wo, total_k] NHWK */
    const rocke_tensor_descriptor_t* dy_desc;
    {
        static const char* const dy_coords[4] = {"n", "ho", "wo", "k"};
        int dy_len[4];
        dy_len[0] = p->N;
        dy_len[1] = Ho;
        dy_len[2] = Wo;
        dy_len[3] = total_k;
        dy_desc = rocke_tensor_descriptor_naive(b, "A", dy_len, 4, NULL, dy_coords, 4);
    }
    /* W descriptor: B[total_k, KH, KW, cpg] KRSC */
    const rocke_tensor_descriptor_t* b_desc;
    {
        static const char* const b_coords[4] = {"k", "r", "s", "c"};
        int b_len[4];
        b_len[0] = total_k;
        b_len[1] = p->KH;
        b_len[2] = p->KW;
        b_len[3] = p->cpg;
        b_desc = rocke_tensor_descriptor_naive(b, "B", b_len, 4, NULL, b_coords, 4);
    }
    /* dX descriptor: D[N, H, W, total_c] NHWC */
    const rocke_tensor_descriptor_t* d_desc;
    {
        static const char* const d_coords[4] = {"n", "h", "w", "c"};
        int d_len[4];
        d_len[0] = p->N;
        d_len[1] = p->H;
        d_len[2] = p->W;
        d_len[3] = total_c;
        d_desc = rocke_tensor_descriptor_naive(b, "D", d_len, 4, NULL, d_coords, 4);
    }

    /* Stride between consecutive k_out values in W:
     * W[k+1, r, s, c] - W[k, r, s, c] = KH*KW*cpg * 2 bytes */
    rocke_value_t* k_stride_bytes = rocke_b_const_i32(b, p->KH * p->KW * p->cpg * 2);
    rocke_value_t* c_Wi = rocke_b_const_i32(b, p->W);
    rocke_value_t* c_kpg = rocke_b_const_i32(b, p->kpg);

    /* Python line 3630: c_st = b.const_i32(stride) if stride > 1 else None
     * Emitted once here, before the Hi-loop, so every tap_valid call reuses it. */
    rocke_value_t* c_st_hw = (c_stride > 1) ? rocke_b_const_i32(b, c_stride) : NULL;

    /* Runtime Hi-loop */
    rocke_value_t* hi_bound = rocke_b_const_i32(b, p->H);
    rocke_iter_arg_t hi_iarg;
    hi_iarg.name = "dg_hi_dummy";
    hi_iarg.init = rocke_b_const_i32(b, 0);
    rocke_for_t hi_loop = rocke_b_scf_for_iter(b,
                                               c0,
                                               hi_bound,
                                               c1,
                                               &hi_iarg,
                                               1,
                                               "dg_hi",
                                               /*unroll=*/false,
                                               /*elide_trailing_barrier=*/false);
    rocke_b_region_enter(b, hi_loop.body);

    rocke_value_t* hi_iv = hi_loop.iv;
    rocke_value_t* dummy_in = hi_loop.iter_vars[0];

    for(j = 0; j < BLOCK_W; j++)
    {
        rocke_value_t* wi = rocke_b_add(b, wi_tile_start, rocke_b_const_i32(b, j));
        rocke_value_t* wi_ok = rocke_b_cmp_lt(b, wi, c_Wi);

        rocke_value_t* acc = zero_f32;

        for(r_const = 0; r_const < p->KH; r_const++)
        {
            rocke_value_t* hi_p_r = rocke_b_add(b, hi_iv, rocke_b_const_i32(b, p->PAD - r_const));
            rocke_value_t* ho;
            rocke_value_t* r_valid;
            tap_valid_h(b, hi_p_r, c0, c_st_hw, c_stride, Ho, &ho, &r_valid);

            for(s_const = 0; s_const < p->KW; s_const++)
            {
                rocke_value_t* wi_p_s = rocke_b_add(b, wi, rocke_b_const_i32(b, p->PAD - s_const));
                rocke_value_t* wo;
                rocke_value_t* s_valid;
                tap_valid_w(b, wi_p_s, c0, c_st_hw, c_stride, Wo, &wo, &s_valid);

                /* Python: land(land(r_valid, s_valid), land(c_in_ok, wi_ok))
                 * Emit in Python order: r_s first, then c_wi, then outer land */
                rocke_value_t* r_s_and = rocke_b_land(b, r_valid, s_valid);
                rocke_value_t* c_wi_and = rocke_b_land(b, c_in_ok, wi_ok);
                rocke_value_t* tap_valid = rocke_b_land(b, r_s_and, c_wi_and);

                /* c_in_in_grp = c_in % cpg */
                rocke_value_t* c_in_in_grp = rocke_b_mod(b, c_in, rocke_b_const_i32(b, p->cpg));
                /* grp = c_in / cpg, k_base = grp * kpg */
                rocke_value_t* grp = rocke_b_div(b, c_in, rocke_b_const_i32(b, p->cpg));
                rocke_value_t* k_base = rocke_b_mul(b, grp, c_kpg);

                /* W base offset at (k=k_base, r, s, c_in_in_grp) in bytes */
                const char* w_names[4] = {"k", "r", "s", "c"};
                rocke_value_t* w_vals[4];
                w_vals[0] = k_base;
                w_vals[1] = rocke_b_const_i32(b, r_const);
                w_vals[2] = rocke_b_const_i32(b, s_const);
                w_vals[3] = c_in_in_grp;
                rocke_value_t* w_off0 = NULL;
                rocke_transforms_descriptor_offset(b, b_desc, w_names, w_vals, 4, &w_off0, NULL);
                rocke_value_t* w_off0_bytes = rocke_b_mul(b, w_off0, c_half_bytes);

                /* dY base offset at (n, ho, wo, k=k_base) in bytes */
                const char* dy_names[4] = {"n", "ho", "wo", "k"};
                rocke_value_t* dy_vals[4];
                dy_vals[0] = n;
                dy_vals[1] = ho;
                dy_vals[2] = wo;
                dy_vals[3] = k_base;
                rocke_value_t* dy_off0 = NULL;
                rocke_transforms_descriptor_offset(
                    b, dy_desc, dy_names, dy_vals, 4, &dy_off0, NULL);
                rocke_value_t* dy_off0_bytes = rocke_b_mul(b, dy_off0, c_half_bytes);

                /* Inner loop over k_out within the group */
                char loop_tag[64];
                snprintf(loop_tag, sizeof(loop_tag), "dg_rs_r%d_s%d_j%d", r_const, s_const, j);
                char k_acc_name[80];
                snprintf(k_acc_name, sizeof(k_acc_name), "k_acc_%s", loop_tag);
                char k_iv_name[80];
                snprintf(k_iv_name, sizeof(k_iv_name), "dg_k_%s", loop_tag);

                rocke_iter_arg_t k_iarg;
                k_iarg.name = k_acc_name;
                k_iarg.init = acc;
                rocke_for_t k_loop = rocke_b_scf_for_iter(b,
                                                          c0,
                                                          c_kpg,
                                                          c1,
                                                          &k_iarg,
                                                          1,
                                                          k_iv_name,
                                                          /*unroll=*/false,
                                                          /*elide_trailing_barrier=*/false);
                rocke_b_region_enter(b, k_loop.body);

                rocke_value_t* k_iv = k_loop.iv;
                rocke_value_t* acc_k = k_loop.iter_vars[0];

                /* W[k_base + k_iv, r, s, c_in_in_grp]: strided k_out access */
                rocke_value_t* k_byte_off = rocke_b_mul(b, k_iv, k_stride_bytes);
                rocke_value_t* w_byte = rocke_b_add(b, w_off0_bytes, k_byte_off);
                rocke_value_t* safe_w = rocke_b_select(b, tap_valid, w_byte, oob_sentinel);
                rocke_value_t* w_h = is_bf16 ? rocke_b_buffer_load_bf16(b, b_rsrc, safe_w, c0)
                                             : rocke_b_buffer_load_f16(b, b_rsrc, safe_w, c0);
                rocke_value_t* w_cast = rocke_b_cast_to_f32(b, w_h);
                rocke_value_t* w_f32 = rocke_b_select(b, tap_valid, w_cast, zero_f32);

                /* dY[n, ho, wo, k_base + k_iv] */
                rocke_value_t* dy_byte
                    = rocke_b_add(b, dy_off0_bytes, rocke_b_mul(b, k_iv, c_half_bytes));
                rocke_value_t* safe_dy = rocke_b_select(b, tap_valid, dy_byte, oob_sentinel);
                rocke_value_t* dy_h = is_bf16 ? rocke_b_buffer_load_bf16(b, a_rsrc, safe_dy, c0)
                                              : rocke_b_buffer_load_f16(b, a_rsrc, safe_dy, c0);
                rocke_value_t* dy_cast = rocke_b_cast_to_f32(b, dy_h);
                rocke_value_t* dy_f32 = rocke_b_select(b, tap_valid, dy_cast, zero_f32);

                rocke_value_t* new_acc = rocke_b_fma(b, w_f32, dy_f32, acc_k);
                rocke_value_t* k_yield[1];
                k_yield[0] = new_acc;
                rocke_b_scf_yield(b, k_yield, 1);
                rocke_b_region_leave(b);

                acc = k_loop.op->results[0];
            }
        }

        /* Store dX[n, hi, wi, c_in] */
        const char* d_names[4] = {"n", "h", "w", "c"};
        rocke_value_t* d_vals[4];
        d_vals[0] = n;
        d_vals[1] = hi_iv;
        d_vals[2] = wi;
        d_vals[3] = c_in;
        rocke_value_t* d_off = NULL;
        rocke_transforms_descriptor_offset(b, d_desc, d_names, d_vals, 4, &d_off, NULL);
        /* Python: land(c_in_ok, wi_ok) — c_in_ok first */
        rocke_value_t* store_guard = rocke_b_land(b, c_in_ok, wi_ok);
        rocke_value_t* d_bytes = rocke_b_mul(b, d_off, c_half_bytes);
        rocke_value_t* safe_d = rocke_b_select(b, store_guard, d_bytes, oob_sentinel);
        rocke_value_t* acc_h
            = is_bf16 ? rocke_b_trunc_f32_to_bf16(b, acc) : rocke_b_trunc_f32_to_f16(b, acc);
        if(is_bf16)
            rocke_b_buffer_store_bf16(b, d_rsrc, safe_d, c0, acc_h);
        else
            rocke_b_buffer_store_f16(b, d_rsrc, safe_d, c0, acc_h);
    }

    rocke_value_t* hi_yield[1];
    hi_yield[0] = dummy_in;
    rocke_b_scf_yield(b, hi_yield, 1);
    rocke_b_region_leave(b);

    (void)c_total_k; /* emitted above; unused after */
    return rocke_ir_builder_kernel(b);
}

/* ===================================================================== *
 *  build_direct_depthwise_dgrad
 *
 *  Ports Python lines 3848-3994 (build_direct_depthwise_dgrad).
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    int r_const, s_const, j;

    if(rocke_direct_depthwise_dgrad_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }
    if(!rocke_direct_depthwise_dgrad_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }

    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int BLOCK_W = spec->block_w;
    const int BLOCK_WAVES = spec->block_waves;
    const int WAVE = spec->wave_size;
    const int THREADS = BLOCK_WAVES * WAVE;
    const int BLOCK_CH = BLOCK_WAVES * WAVE;
    const int Ho = (p->H + 2 * p->PAD - p->KH) / p->stride + 1;
    const int Wo = (p->W + 2 * p->PAD - p->KW) / p->stride + 1;
    const int c_stride = p->stride;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);

    /* Params — io_type = _io_type(p.dtype): f16 or bf16 IR type. */
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    const rocke_type_t* ioptr = rocke_ptr_type(b, io_type, "global");
    const int is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0);
    rocke_param_opts_t ro = {0};
    ro.noalias = true;
    ro.noalias_set = true;
    ro.readonly = true;
    ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;
    rocke_param_opts_t wo_opts = {0};
    wo_opts.noalias = true;
    wo_opts.noalias_set = true;
    wo_opts.writeonly = true;
    wo_opts.writeonly_set = true;
    wo_opts.align = 16;
    wo_opts.align_set = true;
    rocke_param_opts_t none = {0};

    rocke_value_t* A = rocke_b_param(b, "A", ioptr, &ro);
    rocke_value_t* Bp = rocke_b_param(b, "B", ioptr, &ro);
    rocke_value_t* D = rocke_b_param(b, "D", ioptr, &wo_opts);
    rocke_value_t* A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
    rocke_value_t* B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
    rocke_value_t* D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);

    /* Constants — Python order (lines 3873-3879):
     * c0, c1, c_groups, c_half_bytes, c_wave, oob_sentinel, zero_f32 */
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_groups = rocke_b_const_i32(b, p->groups);
    rocke_value_t* c_half_bytes = rocke_b_const_i32(b, 2);
    rocke_value_t* c_wave = rocke_b_const_i32(b, WAVE);
    rocke_value_t* oob_sentinel = rocke_b_const_i32(b, (int32_t)(((int64_t)1 << 31) - 1));
    rocke_value_t* zero_f32 = rocke_b_const_f32(b, 0.0f);

    /* Thread / wave / lane */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* wave_id = rocke_b_div(b, tid, c_wave);
    rocke_value_t* lane = rocke_b_mod(b, tid, c_wave);

    /* Grid: bx=W-tile, by=channel-tile, bz=batch */
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* n = rocke_b_block_id_z(b);

    rocke_value_t* wi_tile_start = rocke_b_mul(b, bx, rocke_b_const_i32(b, BLOCK_W));

    /* ch = by * BLOCK_CH + wave_id * WAVE + lane
     * Python: b.add(b.mul(by, BLOCK_CH), b.add(b.mul(wave_id, c_wave), lane))
     * outer (by * BLOCK_CH) is the LEFT argument → emitted first */
    rocke_value_t* ch_outer = rocke_b_mul(b, by, rocke_b_const_i32(b, BLOCK_CH));
    rocke_value_t* ch_mul = rocke_b_mul(b, wave_id, c_wave);
    rocke_value_t* ch_inner = rocke_b_add(b, ch_mul, lane);
    rocke_value_t* ch = rocke_b_add(b, ch_outer, ch_inner);
    rocke_value_t* ch_in_range = rocke_b_cmp_lt(b, ch, c_groups);

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, A, A_bytes);
    rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, Bp, B_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, D, D_bytes);

    /* dY descriptor: A[N, Ho, Wo, groups] NHWK */
    const rocke_tensor_descriptor_t* dy_desc;
    {
        static const char* const dy_coords[4] = {"n", "ho", "wo", "ch"};
        int dy_len[4];
        dy_len[0] = p->N;
        dy_len[1] = Ho;
        dy_len[2] = Wo;
        dy_len[3] = p->groups;
        dy_desc = rocke_tensor_descriptor_naive(b, "A", dy_len, 4, NULL, dy_coords, 4);
    }
    /* W descriptor: B[groups, KH, KW, 1] KRSC */
    const rocke_tensor_descriptor_t* b_desc;
    {
        static const char* const b_coords[4] = {"k", "r", "s", "c"};
        int b_len[4];
        b_len[0] = p->groups;
        b_len[1] = p->KH;
        b_len[2] = p->KW;
        b_len[3] = 1;
        b_desc = rocke_tensor_descriptor_naive(b, "B", b_len, 4, NULL, b_coords, 4);
    }
    /* dX descriptor: D[N, H, W, groups] NHWC */
    const rocke_tensor_descriptor_t* d_desc;
    {
        static const char* const d_coords[4] = {"n", "h", "w", "ch"};
        int d_len[4];
        d_len[0] = p->N;
        d_len[1] = p->H;
        d_len[2] = p->W;
        d_len[3] = p->groups;
        d_desc = rocke_tensor_descriptor_naive(b, "D", d_len, 4, NULL, d_coords, 4);
    }

    /* Preload W[ch, r, s, 0] into f32 registers (KH * KW per lane).
     * Python order: outer r loop, inner s loop. */
#define ROCKE_DGRAD_DW_MAX_KH 8
#define ROCKE_DGRAD_DW_MAX_KW 8
    if(p->KH > ROCKE_DGRAD_DW_MAX_KH || p->KW > ROCKE_DGRAD_DW_MAX_KW)
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }
    rocke_value_t* weights_f32[ROCKE_DGRAD_DW_MAX_KH * ROCKE_DGRAD_DW_MAX_KW];
    for(r_const = 0; r_const < p->KH; r_const++)
    {
        for(s_const = 0; s_const < p->KW; s_const++)
        {
            const char* w_names[4] = {"k", "r", "s", "c"};
            rocke_value_t* w_vals[4];
            w_vals[0] = ch;
            w_vals[1] = rocke_b_const_i32(b, r_const);
            w_vals[2] = rocke_b_const_i32(b, s_const);
            w_vals[3] = c0;
            rocke_value_t* w_off = NULL;
            rocke_transforms_descriptor_offset(b, b_desc, w_names, w_vals, 4, &w_off, NULL);
            /* Python: select(ch_in_range, mul(w_off, c_half_bytes), oob_sentinel) */
            rocke_value_t* w_bytes = rocke_b_mul(b, w_off, c_half_bytes);
            rocke_value_t* safe_w = rocke_b_select(b, ch_in_range, w_bytes, oob_sentinel);
            rocke_value_t* w_h = is_bf16 ? rocke_b_buffer_load_bf16(b, b_rsrc, safe_w, c0)
                                         : rocke_b_buffer_load_f16(b, b_rsrc, safe_w, c0);
            rocke_value_t* w_cast = rocke_b_cast_to_f32(b, w_h);
            weights_f32[r_const * p->KW + s_const]
                = rocke_b_select(b, ch_in_range, w_cast, zero_f32);
        }
    }

    rocke_value_t* c_Wi = rocke_b_const_i32(b, p->W);

    /* Python: c_st = b.const_i32(stride) if stride > 1 else None
     * Emitted once before the Hi-loop so tap_valid calls reuse it. */
    rocke_value_t* c_st_hw = (c_stride > 1) ? rocke_b_const_i32(b, c_stride) : NULL;

    /* Runtime Hi-loop */
    rocke_value_t* hi_bound = rocke_b_const_i32(b, p->H);
    rocke_iter_arg_t hi_iarg;
    hi_iarg.name = "dg_dw_dummy";
    hi_iarg.init = rocke_b_const_i32(b, 0);
    rocke_for_t hi_loop = rocke_b_scf_for_iter(b,
                                               c0,
                                               hi_bound,
                                               c1,
                                               &hi_iarg,
                                               1,
                                               "dg_dw_hi",
                                               /*unroll=*/false,
                                               /*elide_trailing_barrier=*/false);
    rocke_b_region_enter(b, hi_loop.body);

    rocke_value_t* hi_iv = hi_loop.iv;
    rocke_value_t* dummy_in = hi_loop.iter_vars[0];

    for(j = 0; j < BLOCK_W; j++)
    {
        rocke_value_t* wi = rocke_b_add(b, wi_tile_start, rocke_b_const_i32(b, j));
        rocke_value_t* wi_ok = rocke_b_cmp_lt(b, wi, c_Wi);

        rocke_value_t* acc = zero_f32;

        for(r_const = 0; r_const < p->KH; r_const++)
        {
            rocke_value_t* hi_p_r = rocke_b_add(b, hi_iv, rocke_b_const_i32(b, p->PAD - r_const));
            rocke_value_t* ho;
            rocke_value_t* r_valid;
            tap_valid_h(b, hi_p_r, c0, c_st_hw, c_stride, Ho, &ho, &r_valid);

            for(s_const = 0; s_const < p->KW; s_const++)
            {
                rocke_value_t* wi_p_s = rocke_b_add(b, wi, rocke_b_const_i32(b, p->PAD - s_const));
                rocke_value_t* wo;
                rocke_value_t* s_valid;
                tap_valid_w(b, wi_p_s, c0, c_st_hw, c_stride, Wo, &wo, &s_valid);

                /* Python: land(land(r_valid, s_valid), land(ch_in_range, wi_ok)) */
                rocke_value_t* r_s_and = rocke_b_land(b, r_valid, s_valid);
                rocke_value_t* c_wi_and = rocke_b_land(b, ch_in_range, wi_ok);
                rocke_value_t* valid = rocke_b_land(b, r_s_and, c_wi_and);

                /* dY[n, ho, wo, ch] */
                const char* dy_names[4] = {"n", "ho", "wo", "ch"};
                rocke_value_t* dy_vals[4];
                dy_vals[0] = n;
                dy_vals[1] = ho;
                dy_vals[2] = wo;
                dy_vals[3] = ch;
                rocke_value_t* dy_off = NULL;
                rocke_transforms_descriptor_offset(b, dy_desc, dy_names, dy_vals, 4, &dy_off, NULL);
                /* Python: select(valid, mul(dy_off, c_half_bytes), oob_sentinel) */
                rocke_value_t* dy_bytes = rocke_b_mul(b, dy_off, c_half_bytes);
                rocke_value_t* safe_dy = rocke_b_select(b, valid, dy_bytes, oob_sentinel);
                rocke_value_t* dy_h = is_bf16 ? rocke_b_buffer_load_bf16(b, a_rsrc, safe_dy, c0)
                                              : rocke_b_buffer_load_f16(b, a_rsrc, safe_dy, c0);
                rocke_value_t* dy_cast = rocke_b_cast_to_f32(b, dy_h);
                rocke_value_t* dy_f32 = rocke_b_select(b, valid, dy_cast, zero_f32);
                acc = rocke_b_fma(b, weights_f32[r_const * p->KW + s_const], dy_f32, acc);
            }
        }

        /* Store dX[n, hi, wi, ch] */
        const char* d_names[4] = {"n", "h", "w", "ch"};
        rocke_value_t* d_vals[4];
        d_vals[0] = n;
        d_vals[1] = hi_iv;
        d_vals[2] = wi;
        d_vals[3] = ch;
        rocke_value_t* d_off = NULL;
        rocke_transforms_descriptor_offset(b, d_desc, d_names, d_vals, 4, &d_off, NULL);
        /* Python: land(ch_in_range, wi_ok) — ch_in_range first */
        rocke_value_t* store_guard = rocke_b_land(b, ch_in_range, wi_ok);
        rocke_value_t* d_bytes = rocke_b_mul(b, d_off, c_half_bytes);
        rocke_value_t* safe_d = rocke_b_select(b, store_guard, d_bytes, oob_sentinel);
        rocke_value_t* acc_h
            = is_bf16 ? rocke_b_trunc_f32_to_bf16(b, acc) : rocke_b_trunc_f32_to_f16(b, acc);
        if(is_bf16)
            rocke_b_buffer_store_bf16(b, d_rsrc, safe_d, c0, acc_h);
        else
            rocke_b_buffer_store_f16(b, d_rsrc, safe_d, c0, acc_h);
    }

    rocke_value_t* hi_yield[1];
    hi_yield[0] = dummy_in;
    rocke_b_scf_yield(b, hi_yield, 1);
    rocke_b_region_leave(b);

    return rocke_ir_builder_kernel(b);
}
