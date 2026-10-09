// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_abi.cpp -- the AOT convolution kernarg ABI (C++ twin of the library's
 * kernels/common/conv_abi.py). See instance_conv_abi.h.
 *
 * The list builders below follow the Python module function by function
 * (_dims, _magic, _grid, _fwd_arg_names, ...) so a change to one is easy to
 * mirror in the other; the byte-identity gate compares the kernels built from
 * both.
 */
#include "rocke/instance_conv_abi.h"

#include <stdio.h>
#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

namespace
{

void push(rocke_conv_arg_list_t* out, const char* name, rocke_conv_arg_kind_t kind)
{
    if(out->count >= ROCKE_CONV_MAX_ARGS)
        return; /* capacity is sized for the largest list; never reached */
    rocke_conv_arg_t* a = &out->items[out->count++];
    snprintf(a->name, sizeof(a->name), "%s", name);
    a->kind = kind;
}

void push_i32(rocke_conv_arg_list_t* out, const char* name)
{
    push(out, name, ROCKE_CONV_ARG_I32);
}

/* _dims(is_3d): problem extents, filter extents and the conv attributes. */
void dims(rocke_conv_arg_list_t* out, bool is_3d)
{
    push_i32(out, "p_N");
    push_i32(out, "p_Hi");
    push_i32(out, "p_Wi");
    push_i32(out, "p_C");
    push_i32(out, "p_K");
    push_i32(out, "p_Y");
    push_i32(out, "p_X");
    if(is_3d)
    {
        push_i32(out, "p_Z");
        push_i32(out, "p_Di");
    }
    push_i32(out, "p_sH");
    push_i32(out, "p_sW");
    push_i32(out, "p_pH");
    push_i32(out, "p_pW");
    push_i32(out, "p_dH");
    push_i32(out, "p_dW");
    if(is_3d)
    {
        push_i32(out, "p_sD");
        push_i32(out, "p_pD");
        push_i32(out, "p_dD");
    }
    push_i32(out, "p_groups");
    push_i32(out, "p_Ho");
    push_i32(out, "p_Wo");
    if(is_3d)
        push_i32(out, "p_Do");
    push_i32(out, "p_cpg");
    push_i32(out, "p_kpg");
}

/* _magic(prefix, names): a (mult, shift) pair per divisor, in order. */
void magic(rocke_conv_arg_list_t* out, const char* prefix, const char* const* names, int n)
{
    char buf[ROCKE_CONV_ARG_NAME_CAP];
    for(int i = 0; i < n; ++i)
    {
        snprintf(buf, sizeof(buf), "p_magic_%s%s_mult", prefix, names[i]);
        push_i32(out, buf);
        snprintf(buf, sizeof(buf), "p_magic_%s%s_shift", prefix, names[i]);
        push_i32(out, buf);
    }
}

/* _magic over the 2-D divisors, with the 3-D one prepended when is_3d. */
void magic_2d3d(rocke_conv_arg_list_t* out,
                const char* prefix,
                bool is_3d,
                const char* outer_3d,
                const char* a,
                const char* b)
{
    const char* with_3d[3] = {outer_3d, a, b};
    const char* only_2d[2] = {a, b};
    if(is_3d)
        magic(out, prefix, with_3d, 3);
    else
        magic(out, prefix, only_2d, 2);
}

/* _grid() */
void grid(rocke_conv_arg_list_t* out)
{
    push_i32(out, "p_num_pid_m");
    push_i32(out, "p_num_pid_n");
}

/* Pointer triple + byte-size triple that opens every direction's list. */
void leading(rocke_conv_arg_list_t* out, const char* a, const char* b, const char* d)
{
    char buf[ROCKE_CONV_ARG_NAME_CAP];
    push(out, a, ROCKE_CONV_ARG_A);
    push(out, b, ROCKE_CONV_ARG_B);
    push(out, d, ROCKE_CONV_ARG_D);
    snprintf(buf, sizeof(buf), "%s_bytes", a);
    push_i32(out, buf);
    snprintf(buf, sizeof(buf), "%s_bytes", b);
    push_i32(out, buf);
    snprintf(buf, sizeof(buf), "%s_bytes", d);
    push_i32(out, buf);
}

const int kLeadingArgs = 6;

void fwd_arg_names(rocke_conv_arg_list_t* out, bool is_3d)
{
    leading(out, "A", "B", "D");
    dims(out, is_3d);
    /* K_gemm = [Z*]Y*X*cpg (reduction), M = N*[Do*]Ho*Wo (output spatial). */
    push_i32(out, "p_K_gemm");
    push_i32(out, "p_M");
    /* Row-major strides in elements. */
    push_i32(out, "p_A_stride_n");
    if(is_3d)
        push_i32(out, "p_A_stride_di");
    push_i32(out, "p_A_stride_hi");
    push_i32(out, "p_A_stride_wi");
    push_i32(out, "p_B_stride_k");
    if(is_3d)
        push_i32(out, "p_B_stride_z");
    push_i32(out, "p_B_stride_y");
    push_i32(out, "p_B_stride_x");
    push_i32(out, "p_D_stride_n");
    if(is_3d)
        push_i32(out, "p_D_stride_do");
    push_i32(out, "p_D_stride_ho");
    push_i32(out, "p_D_stride_wo");
    /* m -> (n, [do,] ho, wo): divisors Wo, Ho[, Do]. */
    magic_2d3d(out, "m_", is_3d, "Do", "Ho", "Wo");
    /* k -> ([z,] y, x, c): divisors cpg, X[, Y]. */
    magic_2d3d(out, "k_", is_3d, "Y", "X", "cpg");
    grid(out);
}

void wgrad_arg_names(rocke_conv_arg_list_t* out, bool is_3d, bool two_stage)
{
    leading(out, "dY", "X", "dW");
    dims(out, is_3d);
    push_i32(out, "p_wg_M");
    push_i32(out, "p_wg_N");
    push_i32(out, "p_wg_K");
    push_i32(out, "p_dY_stride_n");
    if(is_3d)
        push_i32(out, "p_dY_stride_do");
    push_i32(out, "p_dY_stride_ho");
    push_i32(out, "p_dY_stride_wo");
    push_i32(out, "p_X_stride_n");
    if(is_3d)
        push_i32(out, "p_X_stride_di");
    push_i32(out, "p_X_stride_hi");
    push_i32(out, "p_X_stride_wi");
    push_i32(out, "p_dW_stride_k");
    if(is_3d)
        push_i32(out, "p_dW_stride_z");
    push_i32(out, "p_dW_stride_y");
    push_i32(out, "p_dW_stride_x");
    /* k_wg -> (n, [do,] ho, wo): divisors Wo, Ho[, Do]. */
    magic_2d3d(out, "k_", is_3d, "Do", "Ho", "Wo");
    /* n_wg -> ([z,] y, x, c): divisors cpg, X[, Y]. */
    magic_2d3d(out, "n_", is_3d, "Y", "X", "cpg");
    grid(out);
    /* ---- variant-specific extras ---- */
    if(two_stage)
    {
        push(out, "ws_ptr", ROCKE_CONV_ARG_F32_PTR);
        push_i32(out, "ws_bytes");
    }
    /* The split-K slice width and degree, always. */
    push_i32(out, "ks");
    push_i32(out, "ks_count");
}

void dgrad_arg_names(rocke_conv_arg_list_t* out, bool is_3d)
{
    leading(out, "dY", "W", "dX");
    dims(out, is_3d);
    push_i32(out, "p_dg_M");
    push_i32(out, "p_dg_N");
    push_i32(out, "p_dg_K");
    push_i32(out, "p_dY_stride_n");
    if(is_3d)
        push_i32(out, "p_dY_stride_do");
    push_i32(out, "p_dY_stride_ho");
    push_i32(out, "p_dY_stride_wo");
    push_i32(out, "p_W_stride_k");
    if(is_3d)
        push_i32(out, "p_W_stride_z");
    push_i32(out, "p_W_stride_y");
    push_i32(out, "p_W_stride_x");
    push_i32(out, "p_dX_stride_n");
    if(is_3d)
        push_i32(out, "p_dX_stride_di");
    push_i32(out, "p_dX_stride_hi");
    push_i32(out, "p_dX_stride_wi");
    /* m -> (n, [di,] hi, wi): divisors Wi, Hi[, Di]. */
    magic_2d3d(out, "m_", is_3d, "Di", "Hi", "Wi");
    grid(out);
    /* The tilde decomposition record buffer is always present. */
    push(out, "sub_gemm_buf", ROCKE_CONV_ARG_I32_PTR);
    push_i32(out, "num_sub_gemms");
}

} // namespace

extern "C" {

const char* rocke_conv_arg_kind_str(rocke_conv_arg_kind_t kind)
{
    switch(kind)
    {
    case ROCKE_CONV_ARG_A:
        return "a";
    case ROCKE_CONV_ARG_B:
        return "b";
    case ROCKE_CONV_ARG_D:
        return "d";
    case ROCKE_CONV_ARG_F32_PTR:
        return "f32*";
    case ROCKE_CONV_ARG_I32_PTR:
        return "i32*";
    case ROCKE_CONV_ARG_I32:
        return "i32";
    }
    return "?";
}

bool rocke_conv_arg_names(const char* direction,
                          bool is_3d,
                          bool two_stage,
                          rocke_conv_arg_list_t* out)
{
    out->count = 0;
    if(direction == NULL)
        return false;
    /* two_stage belongs to wgrad's deterministic epilogue; accepting it
     * elsewhere would describe a kernel the flag cannot build. */
    if(two_stage && strcmp(direction, "wgrad") != 0)
        return false;
    if(strcmp(direction, "fwd") == 0)
        fwd_arg_names(out, is_3d);
    else if(strcmp(direction, "wgrad") == 0)
        wgrad_arg_names(out, is_3d, two_stage);
    else if(strcmp(direction, "dgrad") == 0)
        dgrad_arg_names(out, is_3d);
    else
        return false;
    return true;
}

bool rocke_conv_direct_arg_names(const char* direction, rocke_conv_arg_list_t* out)
{
    out->count = 0;
    const bool is_dgrad = direction != NULL && strcmp(direction, "dgrad") == 0;
    const bool is_wgrad = direction != NULL && strcmp(direction, "wgrad") == 0;
    if(direction == NULL || (!is_dgrad && !is_wgrad && strcmp(direction, "fwd") != 0))
        return false;
    leading(out, "A", "B", "D");
    push_i32(out, "p_N");
    push_i32(out, "p_Hi");
    push_i32(out, "p_Wi");
    push_i32(out, "p_Ho");
    push_i32(out, "p_Wo");
    push_i32(out, "p_groups");
    push_i32(out, "p_total_c"); /* groups * cpg */
    push_i32(out, "p_total_k"); /* groups * kpg */
    /* Strides of the activation-shaped tensors. dgrad runs NHWK -> NHWC, so
     * the two stride triples follow the swapped layouts; wgrad reads dY
     * through A and X through B and writes the filter-shaped dW. */
    if(is_wgrad)
    {
        push_i32(out, "p_dY_stride_n");
        push_i32(out, "p_dY_stride_ho");
        push_i32(out, "p_dY_stride_wo");
        push_i32(out, "p_X_stride_n");
        push_i32(out, "p_X_stride_hi");
        push_i32(out, "p_X_stride_wi");
    }
    else if(is_dgrad)
    {
        push_i32(out, "p_dY_stride_n");
        push_i32(out, "p_dY_stride_ho");
        push_i32(out, "p_dY_stride_wo");
        push_i32(out, "p_dX_stride_n");
        push_i32(out, "p_dX_stride_hi");
        push_i32(out, "p_dX_stride_wi");
    }
    else
    {
        push_i32(out, "p_A_stride_n");
        push_i32(out, "p_A_stride_hi");
        push_i32(out, "p_A_stride_wi");
        push_i32(out, "p_D_stride_n");
        push_i32(out, "p_D_stride_ho");
        push_i32(out, "p_D_stride_wo");
    }
    return true;
}

void rocke_conv_fwd_problem_block(bool is_3d, rocke_conv_arg_list_t* out)
{
    rocke_conv_arg_list_t full;
    full.count = 0;
    fwd_arg_names(&full, is_3d);
    out->count = 0;
    for(int i = kLeadingArgs; i < full.count; ++i)
        push(out, full.items[i].name, full.items[i].kind);
}

bool rocke_conv_emit_param_block(rocke_ir_builder_t* b,
                                 const rocke_conv_arg_list_t* args,
                                 rocke_conv_declare_ptr_fn declare_ptr,
                                 void* user,
                                 const rocke_conv_param_slot_t* slots,
                                 int num_slots)
{
    for(int i = 0; i < args->count; ++i)
    {
        const rocke_conv_arg_t* a = &args->items[i];
        rocke_value_t* v;
        if(a->kind == ROCKE_CONV_ARG_I32)
        {
            v = rocke_b_param(b, a->name, rocke_i32(), NULL);
        }
        else
        {
            if(declare_ptr == NULL)
            {
                rocke_i_set_err(b,
                                ROCKE_ERR_VALUE,
                                "AOT arg %s has kind %s but no declare_ptr callback was "
                                "supplied",
                                a->name,
                                rocke_conv_arg_kind_str(a->kind));
                return false;
            }
            v = declare_ptr(b, a->name, a->kind, user);
        }
        for(int s = 0; s < num_slots; ++s)
        {
            if(strcmp(slots[s].name, a->name) == 0)
            {
                *slots[s].slot = v;
                break;
            }
        }
    }
    return true;
}

} // extern "C"
