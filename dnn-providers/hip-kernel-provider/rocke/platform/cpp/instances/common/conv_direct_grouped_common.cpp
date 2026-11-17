// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_common.cpp -- the AOT plumbing shared by every
 * direct-conv variant.
 *
 * Mirrors the "AOT plumbing shared by every direct-conv variant" section of
 * library/kernels/common/conv_direct_grouped.py:
 *
 *   Python                              C++ (this TU)
 *   ---------------------------------   ------------------------------------
 *   emit_direct_params(b, ...)      rocke_dconv_emit_params
 *   direct_a_descriptor_dynamic(...)    rocke_dconv_a_descriptor_dynamic
 *   direct_d_descriptor_dynamic(...)    rocke_dconv_d_descriptor_dynamic
 *   direct_row_loop_unroll(kh, ...)     rocke_dconv_row_loop_unroll
 */
#include <string.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_abi.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ===================================================================== *
 *  emit_direct_params
 *
 *  Declares the kernarg block in conv_abi order. The order IS the ABI:
 *  kernargs are packed positionally from the launch signature, so a
 *  divergence here does not raise a missing-key error -- it silently shifts
 *  every argument past the first difference.
 * ===================================================================== */
/* Direct conv pointers: D is the output, the rest are inputs. The operand
 * element type rides in `user`. Under wgrad D is the fp32 dW the kernel
 * global_atomic_adds into: an atomic reads its target, so it is not
 * writeonly, and a scalar f32 atomic only needs dword alignment. */
typedef struct dconv_ptr_decl
{
    const rocke_type_t* io_type;
    bool wgrad;
} dconv_ptr_decl_t;

static rocke_value_t* dconv_declare_ptr(rocke_ir_builder_t* b,
                                        const char* name,
                                        rocke_conv_arg_kind_t kind,
                                        void* user)
{
    const dconv_ptr_decl_t* decl = (const dconv_ptr_decl_t*)user;
    rocke_param_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    if(kind == ROCKE_CONV_ARG_D && decl->wgrad)
    {
        opts.align = 4;
        opts.align_set = true;
        return rocke_b_param(b, name, rocke_ptr_type(b, rocke_f32(), "global"), &opts);
    }
    if(kind == ROCKE_CONV_ARG_D)
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
    return rocke_b_param(b, name, rocke_ptr_type(b, decl->io_type, "global"), &opts);
}

void rocke_dconv_emit_params(rocke_ir_builder_t* b,
                             rocke_dconv_params_t* params,
                             const char* direction,
                             const rocke_type_t* io_type)
{
    dconv_ptr_decl_t decl;
    memset(params, 0, sizeof(*params));
    if(direction == NULL)
        direction = "fwd";
    decl.io_type = io_type ? io_type : rocke_f16();
    decl.wgrad = strcmp(direction, "wgrad") == 0;
    /* The A/D stride slots hold whichever tensor the direction binds there:
     * forward NHWC -> NHWK, dgrad dY (NHWK) -> dX (NHWC), wgrad dY (NHWK) in
     * A with X (NHWC) in B. */
    const rocke_conv_param_slot_t slots[] = {
        {"A", &params->A},
        {"B", &params->Bp},
        {"D", &params->D},
        {"A_bytes", &params->A_bytes},
        {"B_bytes", &params->B_bytes},
        {"D_bytes", &params->D_bytes},
        {"p_N", &params->p_N},
        {"p_Hi", &params->p_Hi},
        {"p_Wi", &params->p_Wi},
        {"p_Ho", &params->p_Ho},
        {"p_Wo", &params->p_Wo},
        {"p_groups", &params->p_groups},
        {"p_total_c", &params->p_total_c},
        {"p_total_k", &params->p_total_k},
        {"p_A_stride_n", &params->p_A_stride_n},
        {"p_A_stride_hi", &params->p_A_stride_hi},
        {"p_A_stride_wi", &params->p_A_stride_wi},
        {"p_D_stride_n", &params->p_D_stride_n},
        {"p_D_stride_ho", &params->p_D_stride_ho},
        {"p_D_stride_wo", &params->p_D_stride_wo},
        {"p_dY_stride_n", &params->p_A_stride_n},
        {"p_dY_stride_ho", &params->p_A_stride_hi},
        {"p_dY_stride_wo", &params->p_A_stride_wi},
        {"p_dX_stride_n", &params->p_D_stride_n},
        {"p_dX_stride_hi", &params->p_D_stride_ho},
        {"p_dX_stride_wi", &params->p_D_stride_wo},
        {"p_X_stride_n", &params->p_B_stride_n},
        {"p_X_stride_hi", &params->p_B_stride_hi},
        {"p_X_stride_wi", &params->p_B_stride_wi},
    };
    rocke_conv_arg_list_t abi;
    if(!rocke_conv_direct_arg_names(direction, &abi))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "direct conv has no %s kernel", direction);
        return;
    }
    rocke_conv_emit_param_block(
        b, &abi, dconv_declare_ptr, &decl, slots, (int)(sizeof(slots) / sizeof(slots[0])));
}

/* ===================================================================== *
 *  direct_a_descriptor_dynamic
 *
 *  A[N, H, W, total_c] with runtime extents. Two embeds fold the
 *  conv-spatial algebra in, exactly as the compile-time version did; only
 *  the bounds and the base strides become kernargs. PAD and stride stay
 *  build-time -- they shape the LDS row and the filter-tap offsets.
 * ===================================================================== */
struct rocke_dynamic_tensor_descriptor*
    rocke_dconv_a_descriptor_dynamic(rocke_ir_builder_t* b,
                                     const rocke_dconv_params_t* params,
                                     int pad,
                                     int stride,
                                     const char* w_upper_0,
                                     const char* w_upper_1)
{
    const char* coord_names[4] = {"n", "h", "w", "c"};
    rocke_value_t* strides[4];
    rocke_dynamic_tensor_descriptor_t* desc;

    strides[0] = params->p_A_stride_n;
    strides[1] = params->p_A_stride_hi;
    strides[2] = params->p_A_stride_wi;
    strides[3] = rocke_b_const_i32(b, 1);

    desc = rocke_tensor_descriptor_naive_dynamic(b, "A_nhwc_direct", coord_names, 4, strides);
    if(!desc)
        return NULL;

    {
        const rocke_transform_t* xforms[2];
        rocke_tensor_descriptor_t* chained;

        /* embed(("y_iter",) -> "h", strides=(1,), offset=-PAD, lo=0, hi=Hi) */
        {
            const char* h_upper[1] = {"y_iter"};
            rocke_value_t* h_strides[1] = {NULL};
            const int h_strides_c[1] = {1};
            xforms[0] = rocke_embed_dynamic_mixed(
                b, h_upper, 1, "h", h_strides, h_strides_c, NULL, -pad, 0, params->p_Hi);
            if(!xforms[0])
                return NULL;
        }
        /* embed(w_upper -> "w", strides=(stride, 1), offset=-PAD, lo=0, hi=Wi) */
        {
            const char* w_upper[2] = {w_upper_0, w_upper_1};
            rocke_value_t* w_strides[2] = {NULL, NULL};
            const int w_strides_c[2] = {stride, 1};
            xforms[1] = rocke_embed_dynamic_mixed(
                b, w_upper, 2, "w", w_strides, w_strides_c, NULL, -pad, 0, params->p_Wi);
            if(!xforms[1])
                return NULL;
        }

        chained = rocke_tensor_descriptor_transform(b, &desc->base, xforms, 2);
        if(!chained)
            return NULL;
        desc->base = *chained;
    }
    return desc;
}

/* ===================================================================== *
 *  direct_d_descriptor_dynamic -- D[N, Ho, Wo, total_k], no transforms.
 * ===================================================================== */
struct rocke_dynamic_tensor_descriptor*
    rocke_dconv_d_descriptor_dynamic(rocke_ir_builder_t* b, const rocke_dconv_params_t* params)
{
    const char* coord_names[4] = {"n", "h", "w", "k"};
    rocke_value_t* strides[4];

    strides[0] = params->p_D_stride_n;
    strides[1] = params->p_D_stride_ho;
    strides[2] = params->p_D_stride_wo;
    strides[3] = rocke_b_const_i32(b, 1);

    return rocke_tensor_descriptor_naive_dynamic(b, "D_nhwk_direct", coord_names, 4, strides);
}

/* ===================================================================== *
 *  direct_row_loop_unroll
 *
 *  The row loop has build-time-periodic pieces that a runtime induction
 *  variable would otherwise break: the accumulator slot (y - r) % KH rotates
 *  with period KH, and where the variant stages rows through LDS the
 *  ping-pong buffer alternates with period 2 -- and an LDS allocation is a
 *  build-time SSA value that cannot be selected by a runtime index.
 *  Unrolling by the lcm lines both up with the body boundary, so every
 *  buffer choice and slot index inside is a constant again while y stays
 *  runtime.
 * ===================================================================== */
int rocke_dconv_row_loop_unroll(int kh, bool lds_ping_pong)
{
    if(!lds_ping_pong)
        return kh;
    return (kh % 2 == 0) ? kh : 2 * kh;
}
