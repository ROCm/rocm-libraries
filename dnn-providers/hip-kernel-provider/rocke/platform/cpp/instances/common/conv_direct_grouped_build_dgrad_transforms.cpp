// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_dgrad_transforms.cpp -- C99 port of the MFMA dgrad
 * weight transforms and the build entry of the parametric direct kernel
 * (library/kernels/common/conv_direct_grouped.py).
 *
 *   Python (conv_direct_grouped.py)              C99 (this file)
 *   build_direct_transpose_weights_dgrad(...) -> rocke_build_direct_transpose_weights_dgrad_new()
 *   build_direct_reorganize_weights(...)      -> rocke_build_direct_reorganize_weights_new()
 *   _reorganize_weights_rtc_body(...)         -> dconv_reorganize_rtc_body()
 *   IRBuilder(spec.kernel_name()) + build_direct_conv -> rocke_build_direct_conv_new()
 *
 * The transforms take only the source/destination pointers and their sizes;
 * with runtime_channels two i32 kernargs (p_cpg, p_kpg) carry the original
 * problem's per-group channel counts. Every helper emits in Python evaluation
 * order (see conv_direct_grouped_build_generic.cpp).
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* The A (readonly) / D (writeonly) operand pointers of a weight transform. */
static rocke_value_t* dconv_xform_ptr(rocke_ir_builder_t* b,
                                      const char* name,
                                      const rocke_type_t* io_type,
                                      bool write)
{
    rocke_param_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    if(write)
    {
        opts.writeonly = true;
        opts.writeonly_set = true;
    }
    else
    {
        opts.readonly = true;
        opts.readonly_set = true;
    }
    opts.align = 16;
    opts.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, io_type, "global"), &opts);
}

/* Shared prologue: A, D, A_bytes, D_bytes [, p_cpg, p_kpg]. */
typedef struct dconv_xform_params
{
    rocke_value_t* A;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* D_bytes;
    rocke_value_t* rtc_cpg;
    rocke_value_t* rtc_kpg;
} dconv_xform_params_t;

static void dconv_xform_params(rocke_ir_builder_t* b,
                               const rocke_type_t* io_type,
                               bool rtc,
                               dconv_xform_params_t* out)
{
    memset(out, 0, sizeof(*out));
    out->A = dconv_xform_ptr(b, "A", io_type, false);
    out->D = dconv_xform_ptr(b, "D", io_type, true);
    out->A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), NULL);
    out->D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), NULL);
    if(rtc)
    {
        /* DIRECT_WEIGHT_TRANSFORM_RTC_ARGS */
        out->rtc_cpg = rocke_b_param(b, "p_cpg", rocke_i32(), NULL);
        out->rtc_kpg = rocke_b_param(b, "p_kpg", rocke_i32(), NULL);
    }
}

static bool dconv_xform_is_bf16(const rocke_direct_conv_problem_t* p)
{
    return p->dtype != NULL && strcmp(p->dtype, "bf16") == 0;
}

/* ===================================================================== *
 *  build_direct_transpose_weights_dgrad
 * ===================================================================== */
static rocke_kernel_def_t*
    dconv_build_transpose(rocke_ir_builder_t* b,
                          const rocke_direct_transpose_weights_dgrad_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool RTC = spec->runtime_channels;
    const bool is_bf16 = dconv_xform_is_bf16(p);
    const int BLOCK = 64;
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, p->dtype);
    dconv_xform_params_t prm;
    if(io_type == NULL)
        return NULL;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BLOCK);
    dconv_xform_params(b, io_type, RTC, &prm);

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c_half_bytes = rocke_b_const_i32(b, 2);
    rocke_value_t* oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* bz = rocke_b_block_id_z(b);

    const int n_rs = p->KH * p->KW;
    (void)rocke_b_const_i32(b, p->KH); /* c_KH (unused) */
    rocke_value_t* c_KW = rocke_b_const_i32(b, p->KW);
    rocke_value_t* c_n_rs = rocke_b_const_i32(b, n_rs);

    rocke_value_t* grp = rocke_b_div(b, bx, c_n_rs);
    rocke_value_t* rs = rocke_b_mod(b, bx, c_n_rs);
    rocke_value_t* r_prime = rocke_b_div(b, rs, c_KW);
    rocke_value_t* s_prime = rocke_b_mod(b, rs, c_KW);
    rocke_value_t* r_flip = rocke_b_sub(b, rocke_b_const_i32(b, p->KH - 1), r_prime);
    rocke_value_t* s_flip = rocke_b_sub(b, rocke_b_const_i32(b, p->KW - 1), s_prime);

    rocke_value_t* k_in_g;
    {
        rocke_value_t* m = rocke_b_mul(b, by, rocke_b_const_i32(b, BLOCK));
        k_in_g = rocke_b_add(b, m, tid);
    }
    rocke_value_t* c_in_g = bz;

    rocke_value_t* k_abs;
    rocke_value_t* c_abs;
    rocke_value_t* k_ok;
    rocke_value_t* c_ok;
    rocke_value_t* c_kpg = prm.rtc_kpg;
    rocke_value_t* c_cpg = prm.rtc_cpg;
    if(RTC)
    {
        k_abs = rocke_b_add(b, rocke_b_mul(b, grp, c_kpg), k_in_g);
        c_abs = rocke_b_add(b, rocke_b_mul(b, grp, c_cpg), c_in_g);
        k_ok = rocke_b_cmp_lt(b, k_in_g, c_kpg);
        c_ok = rocke_b_cmp_lt(b, c_in_g, c_cpg);
    }
    else
    {
        k_abs = rocke_b_add(b, rocke_b_mul(b, grp, rocke_b_const_i32(b, p->kpg)), k_in_g);
        c_abs = rocke_b_add(b, rocke_b_mul(b, grp, rocke_b_const_i32(b, p->cpg)), c_in_g);
        k_ok = rocke_b_cmp_lt(b, k_in_g, rocke_b_const_i32(b, p->kpg));
        c_ok = rocke_b_cmp_lt(b, c_in_g, rocke_b_const_i32(b, p->cpg));
    }
    rocke_value_t* valid = rocke_b_land(b, k_ok, c_ok);

    static const char* const src_coords[4] = {"k", "r", "s", "c"};
    rocke_tensor_descriptor_t* src_desc;
    if(RTC)
        src_desc = rocke_dconv_krsc_descriptor_dynamic(b, "A", src_coords, c_cpg, p->KH, p->KW);
    else
    {
        int lengths[4];
        lengths[0] = rocke_direct_conv_problem_total_k(p);
        lengths[1] = p->KH;
        lengths[2] = p->KW;
        lengths[3] = p->cpg;
        src_desc = rocke_tensor_descriptor_naive(b, "A", lengths, 4, NULL, src_coords, 4);
    }
    if(src_desc == NULL)
        return NULL;
    rocke_value_t* src_off = NULL;
    rocke_value_t* src_valid = NULL;
    {
        const char* names[4] = {"k", "r", "s", "c"};
        rocke_value_t* values[4];
        values[0] = k_abs;
        values[1] = r_flip;
        values[2] = s_flip;
        values[3] = c_in_g;
        if(!rocke_transforms_descriptor_offset(b, src_desc, names, values, 4, &src_off, &src_valid))
            return NULL;
    }
    rocke_value_t* src_safe
        = rocke_b_select(b, valid, rocke_b_mul(b, src_off, c_half_bytes), oob_sentinel);
    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, prm.A, prm.A_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, prm.D, prm.D_bytes);
    rocke_value_t* val = is_bf16 ? rocke_b_buffer_load_bf16(b, a_rsrc, src_safe, c0)
                                 : rocke_b_buffer_load_f16(b, a_rsrc, src_safe, c0);

    static const char* const dst_coords[4] = {"c", "r", "s", "k"};
    rocke_tensor_descriptor_t* dst_desc;
    if(RTC)
        dst_desc = rocke_dconv_krsc_descriptor_dynamic(b, "D", dst_coords, c_kpg, p->KH, p->KW);
    else
    {
        int lengths[4];
        lengths[0] = rocke_direct_conv_problem_total_c(p);
        lengths[1] = p->KH;
        lengths[2] = p->KW;
        lengths[3] = p->kpg;
        dst_desc = rocke_tensor_descriptor_naive(b, "D", lengths, 4, NULL, dst_coords, 4);
    }
    if(dst_desc == NULL)
        return NULL;
    rocke_value_t* dst_off = NULL;
    rocke_value_t* dst_valid = NULL;
    {
        const char* names[4] = {"c", "r", "s", "k"};
        rocke_value_t* values[4];
        values[0] = c_abs;
        values[1] = r_prime;
        values[2] = s_prime;
        values[3] = k_in_g;
        if(!rocke_transforms_descriptor_offset(b, dst_desc, names, values, 4, &dst_off, &dst_valid))
            return NULL;
    }
    rocke_value_t* dst_safe
        = rocke_b_select(b, valid, rocke_b_mul(b, dst_off, c_half_bytes), oob_sentinel);
    if(is_bf16)
        rocke_b_buffer_store_bf16(b, d_rsrc, dst_safe, c0, val);
    else
        rocke_b_buffer_store_f16(b, d_rsrc, dst_safe, c0, val);

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_ir_builder_kernel(b);
}

/* ===================================================================== *
 *  build_direct_reorganize_weights
 * ===================================================================== */
static rocke_value_t* dconv_xform_load(rocke_ir_builder_t* b,
                                       bool is_bf16,
                                       rocke_value_t* rsrc,
                                       rocke_value_t* voff,
                                       rocke_value_t* c0,
                                       int dwords)
{
    return is_bf16 ? rocke_b_buffer_load_vN_bf16(b, rsrc, voff, c0, dwords)
                   : rocke_b_buffer_load_vN_f16(b, rsrc, voff, c0, dwords);
}

static void dconv_xform_store(rocke_ir_builder_t* b,
                              bool is_bf16,
                              rocke_value_t* rsrc,
                              rocke_value_t* voff,
                              rocke_value_t* c0,
                              rocke_value_t* val,
                              int dwords)
{
    if(is_bf16)
        rocke_b_buffer_store_vN_bf16(b, rsrc, voff, c0, val, dwords);
    else
        rocke_b_buffer_store_vN_f16(b, rsrc, voff, c0, val, dwords);
}

/* _reorganize_weights_rtc_body: the W_coa layout's K-atom / M-tile counts
 * follow the runtime channel counts; lanes past the channel tails store the
 * zero their redirected load returned. */
static rocke_kernel_def_t* dconv_reorganize_rtc_body(rocke_ir_builder_t* b,
                                                     const rocke_direct_conv_problem_t* p,
                                                     bool is_bf16,
                                                     int K_ATOM_SZ,
                                                     int ELEMS_PER_LANE,
                                                     const dconv_xform_params_t* prm,
                                                     rocke_value_t* c0,
                                                     rocke_value_t* c_half_bytes,
                                                     rocke_value_t* oob_sentinel,
                                                     rocke_value_t* tid,
                                                     rocke_value_t* bx)
{
    const int WAVE = 64;
    rocke_value_t* c_cpg = prm->rtc_cpg;
    rocke_value_t* c_kpg = prm->rtc_kpg;
    rocke_value_t* n_k_atoms;
    {
        rocke_value_t* c_r = rocke_b_const_i32(b, K_ATOM_SZ - 1);
        rocke_value_t* s = rocke_b_add(b, c_kpg, c_r);
        rocke_value_t* c_a = rocke_b_const_i32(b, K_ATOM_SZ);
        n_k_atoms = rocke_b_div(b, s, c_a);
    }
    rocke_value_t* n_m_tiles;
    {
        rocke_value_t* c_r = rocke_b_const_i32(b, 15);
        rocke_value_t* s = rocke_b_add(b, c_cpg, c_r);
        rocke_value_t* c_a = rocke_b_const_i32(b, 16);
        n_m_tiles = rocke_b_div(b, s, c_a);
    }
    rocke_value_t* n_am = rocke_b_mul(b, n_k_atoms, n_m_tiles);
    rocke_value_t* n_s_am = rocke_b_mul(b, rocke_b_const_i32(b, p->KW), n_am);
    rocke_value_t* n_per_group = rocke_b_mul(b, rocke_b_const_i32(b, p->KH), n_s_am);
    rocke_value_t* grp = rocke_b_div(b, bx, n_per_group);
    rocke_value_t* rem = rocke_b_mod(b, bx, n_per_group);
    rocke_value_t* r_prime = rocke_b_div(b, rem, n_s_am);
    rocke_value_t* rem2 = rocke_b_mod(b, rem, n_s_am);
    rocke_value_t* s_prime = rocke_b_div(b, rem2, n_am);
    rocke_value_t* rem3 = rocke_b_mod(b, rem2, n_am);
    rocke_value_t* atom_idx = rocke_b_div(b, rem3, n_m_tiles);
    rocke_value_t* m_idx = rocke_b_mod(b, rem3, n_m_tiles);

    rocke_value_t* q_in_lane = rocke_b_mod(b, tid, rocke_b_const_i32(b, 16));
    rocke_value_t* c4 = rocke_b_div(b, tid, rocke_b_const_i32(b, 16));

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, prm->A, prm->A_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, prm->D, prm->D_bytes);

    static const char* const wt_coords[4] = {"k_new", "r", "s", "c_new"};
    rocke_tensor_descriptor_t* wt_desc
        = rocke_dconv_krsc_descriptor_dynamic(b, "A", wt_coords, c_kpg, p->KH, p->KW);
    if(wt_desc == NULL)
        return NULL;
    rocke_value_t* m_row;
    {
        rocke_value_t* m = rocke_b_mul(b, m_idx, rocke_b_const_i32(b, 16));
        m_row = rocke_b_add(b, m, q_in_lane);
    }
    rocke_value_t* k_slice;
    {
        rocke_value_t* m1 = rocke_b_mul(b, atom_idx, rocke_b_const_i32(b, K_ATOM_SZ));
        rocke_value_t* m2 = rocke_b_mul(b, c4, rocke_b_const_i32(b, ELEMS_PER_LANE));
        k_slice = rocke_b_add(b, m1, m2);
    }
    rocke_value_t* src_ok;
    {
        rocke_value_t* lt1 = rocke_b_cmp_lt(b, m_row, c_cpg);
        rocke_value_t* lt2 = rocke_b_cmp_lt(b, k_slice, c_kpg);
        src_ok = rocke_b_land(b, lt1, lt2);
    }
    /* The group selects the W_T row; the column is the K slice alone. */
    rocke_value_t* k_new_val = rocke_b_add(b, rocke_b_mul(b, grp, c_cpg), m_row);
    rocke_value_t* src_off = NULL;
    rocke_value_t* src_valid = NULL;
    {
        const char* names[4] = {"k_new", "r", "s", "c_new"};
        rocke_value_t* values[4];
        values[0] = k_new_val;
        values[1] = r_prime;
        values[2] = s_prime;
        values[3] = k_slice;
        if(!rocke_transforms_descriptor_offset(b, wt_desc, names, values, 4, &src_off, &src_valid))
            return NULL;
    }
    rocke_value_t* safe_src
        = rocke_b_select(b, src_ok, rocke_b_mul(b, src_off, c_half_bytes), oob_sentinel);
    rocke_value_t* val = dconv_xform_load(b, is_bf16, a_rsrc, safe_src, c0, ELEMS_PER_LANE / 2);
    rocke_value_t* dst_off;
    {
        rocke_value_t* m1 = rocke_b_mul(b, bx, rocke_b_const_i32(b, WAVE * ELEMS_PER_LANE));
        rocke_value_t* m2 = rocke_b_mul(b, tid, rocke_b_const_i32(b, ELEMS_PER_LANE));
        dst_off = rocke_b_add(b, m1, m2);
    }
    dconv_xform_store(
        b, is_bf16, d_rsrc, rocke_b_mul(b, dst_off, c_half_bytes), c0, val, ELEMS_PER_LANE / 2);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_ir_builder_kernel(b);
}

static rocke_kernel_def_t*
    dconv_build_reorganize(rocke_ir_builder_t* b,
                           const rocke_direct_reorganize_weights_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool fold_k32 = spec->fold_k32;
    const bool RTC = spec->runtime_channels;
    const bool is_bf16 = dconv_xform_is_bf16(p);
    const int K_ATOM_SZ = fold_k32 ? 32 : 16;
    const int ELEMS_PER_LANE = fold_k32 ? 8 : 4;
    const int N_K_ATOMS = (p->kpg + K_ATOM_SZ - 1) / K_ATOM_SZ;
    const int N_M_TILES = (p->cpg + 15) / 16;
    const int WAVE = 64;
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, p->dtype);
    dconv_xform_params_t prm;
    if(io_type == NULL)
        return NULL;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", WAVE);
    dconv_xform_params(b, io_type, RTC, &prm);

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c_half_bytes = rocke_b_const_i32(b, 2);
    rocke_value_t* oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bx = rocke_b_block_id_x(b);

    if(RTC)
        return dconv_reorganize_rtc_body(b,
                                         p,
                                         is_bf16,
                                         K_ATOM_SZ,
                                         ELEMS_PER_LANE,
                                         &prm,
                                         c0,
                                         c_half_bytes,
                                         oob_sentinel,
                                         tid,
                                         bx);

    const int n_per_group = p->KH * p->KW * N_K_ATOMS * N_M_TILES;
    rocke_value_t* grp = rocke_b_div(b, bx, rocke_b_const_i32(b, n_per_group));
    rocke_value_t* rem = rocke_b_mod(b, bx, rocke_b_const_i32(b, n_per_group));
    const int n_s_am = p->KW * N_K_ATOMS * N_M_TILES;
    rocke_value_t* r_prime = rocke_b_div(b, rem, rocke_b_const_i32(b, n_s_am));
    rocke_value_t* rem2 = rocke_b_mod(b, rem, rocke_b_const_i32(b, n_s_am));
    rocke_value_t* s_prime = rocke_b_div(b, rem2, rocke_b_const_i32(b, N_K_ATOMS * N_M_TILES));
    rocke_value_t* rem3 = rocke_b_mod(b, rem2, rocke_b_const_i32(b, N_K_ATOMS * N_M_TILES));
    rocke_value_t* atom_idx = rocke_b_div(b, rem3, rocke_b_const_i32(b, N_M_TILES));
    rocke_value_t* m_idx = rocke_b_mod(b, rem3, rocke_b_const_i32(b, N_M_TILES));

    rocke_value_t* q_in_lane = rocke_b_mod(b, tid, rocke_b_const_i32(b, 16));
    rocke_value_t* c4 = rocke_b_div(b, tid, rocke_b_const_i32(b, 16));

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, prm.A, prm.A_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, prm.D, prm.D_bytes);

    static const char* const wt_coords[4] = {"k_new", "r", "s", "c_new"};
    rocke_tensor_descriptor_t* wt_desc;
    {
        int lengths[4];
        lengths[0] = p->cpg;
        lengths[1] = p->KH;
        lengths[2] = p->KW;
        lengths[3] = p->kpg;
        wt_desc = rocke_tensor_descriptor_naive(b, "A", lengths, 4, NULL, wt_coords, 4);
        if(wt_desc == NULL)
            return NULL;
    }
    rocke_value_t* k_new_val;
    {
        rocke_value_t* m1 = rocke_b_mul(b, grp, rocke_b_const_i32(b, p->cpg));
        rocke_value_t* m2 = rocke_b_mul(b, m_idx, rocke_b_const_i32(b, 16));
        rocke_value_t* a1 = rocke_b_add(b, m2, q_in_lane);
        k_new_val = rocke_b_add(b, m1, a1);
    }
    rocke_value_t* src_ok;
    {
        rocke_value_t* m1 = rocke_b_mul(b, m_idx, rocke_b_const_i32(b, 16));
        rocke_value_t* a1 = rocke_b_add(b, m1, q_in_lane);
        rocke_value_t* lt1 = rocke_b_cmp_lt(b, a1, rocke_b_const_i32(b, p->cpg));
        rocke_value_t* m2 = rocke_b_mul(b, atom_idx, rocke_b_const_i32(b, K_ATOM_SZ));
        rocke_value_t* m3 = rocke_b_mul(b, c4, rocke_b_const_i32(b, ELEMS_PER_LANE));
        rocke_value_t* a2 = rocke_b_add(b, m2, m3);
        rocke_value_t* lt2 = rocke_b_cmp_lt(b, a2, rocke_b_const_i32(b, p->kpg));
        src_ok = rocke_b_land(b, lt1, lt2);
    }
    /* W_T's last dim is the per-group kpg (the transpose writes
     * W_T[c_abs, r', s', k_in_g]): the group is already in k_new, so the
     * column is the K slice alone. */
    rocke_value_t* c_new_base;
    {
        rocke_value_t* m2 = rocke_b_mul(b, atom_idx, rocke_b_const_i32(b, K_ATOM_SZ));
        rocke_value_t* m3 = rocke_b_mul(b, c4, rocke_b_const_i32(b, ELEMS_PER_LANE));
        c_new_base = rocke_b_add(b, m2, m3);
    }
    rocke_value_t* src_off = NULL;
    rocke_value_t* src_valid = NULL;
    {
        const char* names[4] = {"k_new", "r", "s", "c_new"};
        rocke_value_t* values[4];
        values[0] = k_new_val;
        values[1] = r_prime;
        values[2] = s_prime;
        values[3] = c_new_base;
        if(!rocke_transforms_descriptor_offset(b, wt_desc, names, values, 4, &src_off, &src_valid))
            return NULL;
    }
    rocke_value_t* safe_src
        = rocke_b_select(b, src_ok, rocke_b_mul(b, src_off, c_half_bytes), oob_sentinel);
    rocke_value_t* val = dconv_xform_load(b, is_bf16, a_rsrc, safe_src, c0, ELEMS_PER_LANE / 2);
    rocke_value_t* dst_off;
    {
        rocke_value_t* m1 = rocke_b_mul(b, bx, rocke_b_const_i32(b, WAVE * ELEMS_PER_LANE));
        rocke_value_t* m2 = rocke_b_mul(b, tid, rocke_b_const_i32(b, ELEMS_PER_LANE));
        dst_off = rocke_b_add(b, m1, m2);
    }
    /* Every lane stores, zeros where the source is out of range: the main
     * kernel masks only the dY operand of a partial K-atom. */
    val = rocke_b_select(b, src_ok, val, rocke_b_zero_vec(b, io_type, ELEMS_PER_LANE));
    dconv_xform_store(
        b, is_bf16, d_rsrc, rocke_b_mul(b, dst_off, c_half_bytes), c0, val, ELEMS_PER_LANE / 2);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return rocke_ir_builder_kernel(b);
}

/* ===================================================================== *
 *  Public build entries
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_transpose_weights_dgrad_new(
    rocke_ir_builder_t* b,
    const rocke_direct_transpose_weights_dgrad_spec_t* spec,
    const char* arch)
{
    (void)arch;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char short_buf[128];
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_direct_conv_problem_short(&spec->problem, short_buf, sizeof short_buf) != ROCKE_OK)
            return NULL;
        snprintf(name,
                 sizeof name,
                 "direct_transpose_weights_dgrad_%s%s",
                 short_buf,
                 spec->runtime_channels ? "_rtc" : "");
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return dconv_build_transpose(b, spec);
    });
}

rocke_kernel_def_t* rocke_build_direct_reorganize_weights_new(
    rocke_ir_builder_t* b, const rocke_direct_reorganize_weights_spec_t* spec, const char* arch)
{
    (void)arch;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char short_buf[128];
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        /* Raised before the Python builds its IRBuilder. */
        if(spec->fold_k32 && spec->problem.kpg % 32 != 0)
        {
            rocke_ir_builder_init(b, "direct_reorg_wt_invalid");
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "DirectConvSpec fold_k32 requires cpg to be a multiple of 32 (got %d)",
                            spec->problem.kpg);
        }
        if(rocke_direct_conv_problem_short(&spec->problem, short_buf, sizeof short_buf) != ROCKE_OK)
            return NULL;
        snprintf(name,
                 sizeof name,
                 "direct_reorg_wt%s_%s%s",
                 spec->fold_k32 ? "32" : "",
                 short_buf,
                 spec->runtime_channels ? "_rtc" : "");
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return dconv_build_reorganize(b, spec);
    });
}

rocke_kernel_def_t* rocke_build_direct_conv_new(rocke_ir_builder_t* b,
                                                const rocke_direct_conv_spec_t* spec,
                                                const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_direct_conv_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_conv(b, spec, arch);
    });
}
