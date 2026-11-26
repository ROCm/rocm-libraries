// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_bias.cpp -- per-output-channel bias of the convolution forward kernels
 * (C++ twin of the library's kernels/conv/conv_bias.py). See
 * instance_conv_bias.h.
 */
#include "rocke/instance_conv_bias.h"

#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include <string.h>

/* The widest MMA fragment grid a conv kernel builds (mfmas_n per warp). */
#define ROCKE_CONV_BIAS_MAX_NI 64
/* The longest fp32 accumulator fragment (32x32 MFMA: 16 slots). */
#define ROCKE_CONV_BIAS_MAX_SLOTS 64

extern "C" {

void rocke_conv_bias_declare_params(rocke_ir_builder_t* b,
                                    const rocke_type_t* elem,
                                    rocke_value_t** bias,
                                    rocke_value_t** bias_bytes)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    *bias = rocke_b_param(b, "Bias", rocke_ptr_type(b, elem, "global"), &o);
    *bias_bytes = rocke_b_param(b, "Bias_bytes", rocke_i32(), NULL);
}

rocke_value_t*
    rocke_conv_bias_rsrc(rocke_ir_builder_t* b, rocke_value_t* bias, rocke_value_t* bias_bytes)
{
    return rocke_b_buffer_rsrc(b, bias, bias_bytes);
}

/* Bytes of one bias element: the output dtypes a conv forward stores. Any
 * other dtype is an error (2 keeps the emission going; the builder fails). */
static int conv_bias_elem_bytes(rocke_ir_builder_t* b, const rocke_type_t* elem)
{
    if(strcmp(elem->name, "f32") == 0)
        return 4;
    if(strcmp(elem->name, "f16") != 0 && strcmp(elem->name, "bf16") != 0)
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "conv bias: no %s bias (f16, bf16 or f32 only)", elem->name);
    return 2;
}

rocke_value_t* rocke_conv_bias_load_f32(rocke_ir_builder_t* b,
                                        rocke_value_t* rsrc,
                                        rocke_value_t* k,
                                        const rocke_type_t* elem,
                                        rocke_value_t* c0)
{
    const int eb = conv_bias_elem_bytes(b, elem);
    rocke_value_t* c_eb = rocke_b_const_i32(b, eb);
    rocke_value_t* off = rocke_b_mul(b, k, c_eb);
    rocke_value_t* v = rocke_b_buffer_load(b, rsrc, off, c0, elem);
    return rocke_b_cast_to_f32(b, v);
}

rocke_value_t* rocke_conv_bias_load_vec_f32(rocke_ir_builder_t* b,
                                            rocke_value_t* rsrc,
                                            rocke_value_t* k,
                                            int n,
                                            const rocke_type_t* elem,
                                            rocke_value_t* c0)
{
    const int eb = conv_bias_elem_bytes(b, elem);
    const int per = n < 16 / eb ? n : 16 / eb;
    rocke_value_t* comps[ROCKE_CONV_BIAS_MAX_SLOTS];
    int count = 0;
    if(n <= 0 || n > ROCKE_CONV_BIAS_MAX_SLOTS || n % per != 0)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "load_conv_bias_vec_f32: %d elements in loads of %d", n, per);
        return NULL;
    }
    rocke_value_t* c_eb = rocke_b_const_i32(b, eb);
    rocke_value_t* off = rocke_b_mul(b, k, c_eb);
    for(int j = 0; j < n; j += per)
    {
        rocke_value_t* voff = off;
        if(j != 0)
        {
            rocke_value_t* c_j = rocke_b_const_i32(b, j * eb);
            voff = rocke_b_add(b, off, c_j);
        }
        rocke_value_t* v = rocke_b_buffer_load_vN(b, rsrc, voff, c0, elem, per);
        for(int i = 0; i < per; ++i)
        {
            rocke_value_t* e = rocke_b_vec_extract(b, v, i);
            comps[count++] = rocke_b_cast_to_f32(b, e);
        }
    }
    return rocke_b_vec_pack(b, comps, count, rocke_f32());
}

rocke_value_t* rocke_conv_bias_load_run_f32(rocke_ir_builder_t* b,
                                            rocke_value_t* rsrc,
                                            rocke_value_t* k,
                                            int n,
                                            const rocke_type_t* elem,
                                            rocke_value_t* c0)
{
    rocke_value_t* comps[ROCKE_CONV_BIAS_MAX_SLOTS];
    if(n <= 0 || n > ROCKE_CONV_BIAS_MAX_SLOTS)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "load_conv_bias_run_f32: %d elements", n);
        return NULL;
    }
    for(int j = 0; j < n; ++j)
    {
        rocke_value_t* k_j = k;
        if(j != 0)
        {
            rocke_value_t* c_j = rocke_b_const_i32(b, j);
            k_j = rocke_b_add(b, k, c_j);
        }
        comps[j] = rocke_conv_bias_load_f32(b, rsrc, k_j, elem, c0);
    }
    return rocke_b_vec_pack(b, comps, n, rocke_f32());
}

void rocke_conv_bias_add_to_accs(rocke_ir_builder_t* b,
                                 rocke_value_t* const* accs,
                                 rocke_value_t* const* bias_per_ni,
                                 int mfmas_m,
                                 int mfmas_n,
                                 rocke_value_t** out)
{
    rocke_value_t* splat[ROCKE_CONV_BIAS_MAX_NI];
    bool fits = mfmas_n > 0 && mfmas_n <= ROCKE_CONV_BIAS_MAX_NI;
    for(int ni = 0; fits && ni < mfmas_n; ++ni)
        fits = accs[ni]->type->count <= ROCKE_CONV_BIAS_MAX_SLOTS;
    if(!fits)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "add_conv_bias_to_accs: a %dx%d fragment grid past %d columns of "
                        "%d slots",
                        mfmas_m,
                        mfmas_n,
                        ROCKE_CONV_BIAS_MAX_NI,
                        ROCKE_CONV_BIAS_MAX_SLOTS);
        for(int i = 0; i < mfmas_m * mfmas_n; ++i)
            out[i] = accs[i];
        return;
    }
    for(int ni = 0; ni < mfmas_n; ++ni)
    {
        rocke_value_t* elems[ROCKE_CONV_BIAS_MAX_SLOTS];
        const int count = accs[ni]->type->count;
        for(int i = 0; i < count; ++i)
            elems[i] = bias_per_ni[ni];
        splat[ni] = rocke_b_vec_pack(b, elems, count, rocke_f32());
    }
    for(int i = 0; i < mfmas_m * mfmas_n; ++i)
        out[i] = rocke_b_fadd(b, accs[i], splat[i % mfmas_n]);
}

void rocke_conv_bias_apply_implicit_gemm(rocke_ir_builder_t* b,
                                         rocke_value_t* const* accs,
                                         const rocke_mmaop_t* op,
                                         const rocke_type_t* elem,
                                         rocke_value_t* bias,
                                         rocke_value_t* bias_bytes,
                                         rocke_value_t* lane,
                                         rocke_value_t* warp_n_idx,
                                         rocke_value_t* block_n_off,
                                         rocke_value_t* k_out_group_base,
                                         int mfmas_m,
                                         int mfmas_n,
                                         int warp_tile_n,
                                         rocke_value_t** out)
{
    rocke_value_t* vals[ROCKE_CONV_BIAS_MAX_NI];
    if(mfmas_n <= 0 || mfmas_n > ROCKE_CONV_BIAS_MAX_NI)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "apply_implicit_gemm_conv_bias: %d fragment columns (at most %d)",
                        mfmas_n,
                        ROCKE_CONV_BIAS_MAX_NI);
        for(int i = 0; i < mfmas_m * mfmas_n; ++i)
            out[i] = accs[i];
        return;
    }
    rocke_value_t* rsrc = rocke_conv_bias_rsrc(b, bias, bias_bytes);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* row = NULL;
    rocke_value_t* col = NULL;
    rocke_arch_layout_map_coord(rocke_mma_op_dst_layout(op, b), b, lane, 0, &row, &col);
    (void)row;
    rocke_value_t* c_wn = rocke_b_const_i32(b, mfmas_n * warp_tile_n);
    rocke_value_t* warp_n_off = rocke_b_mul(b, warp_n_idx, c_wn);
    rocke_value_t* k = rocke_b_add(b, rocke_b_add(b, block_n_off, warp_n_off), col);
    if(k_out_group_base != NULL)
        k = rocke_b_add(b, k_out_group_base, k);
    for(int ni = 0; ni < mfmas_n; ++ni)
    {
        rocke_value_t* k_ni = k;
        if(ni != 0)
        {
            rocke_value_t* c_ni = rocke_b_const_i32(b, ni * warp_tile_n);
            k_ni = rocke_b_add(b, k, c_ni);
        }
        vals[ni] = rocke_conv_bias_load_f32(b, rsrc, k_ni, elem, c0);
    }
    rocke_conv_bias_add_to_accs(b, accs, vals, mfmas_m, mfmas_n, out);
}

} // extern "C"
