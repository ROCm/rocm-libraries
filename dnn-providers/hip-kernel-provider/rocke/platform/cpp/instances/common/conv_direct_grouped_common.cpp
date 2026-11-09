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
#include <ctype.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.helpers.fuse.h" /* rocke_fuse_dtype_to_ir_str */
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_abi.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include <stdio.h>

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

/* ===================================================================== *
 *  fp32 / TF32 storage helpers.
 *
 *    Python (conv_direct_grouped.py)        C++ (here)
 *    ------------------------------------   ---------------------------------
 *    _storage_dtype(dtype)                  rocke_dconv_storage_dtype
 *    _is_f32_storage(dtype)                 rocke_dconv_is_f32_storage
 *    _elem_bytes(dtype)                     rocke_dconv_elem_bytes
 *    _io_type(dtype)                        rocke_dconv_io_type
 *    _mfma_dtype_flags(dtype)               rocke_dconv_mfma_dtype_flags
 *    _load_elems(b, dtype, ...)             rocke_dconv_load_elems
 *    _store_from_f32(b, dtype, ...)         rocke_dconv_store_from_f32
 *    _store_elems(b, dtype, ...)            rocke_dconv_store_elems
 *    _smem_load_elems(b, dtype, ...)        rocke_dconv_smem_load_elems
 *    _smem_store_elems(b, dtype, ...)       rocke_dconv_smem_store_elems
 *    _f32_mma_plan(arch, dtype, m, frag)    rocke_dconv_f32_mma_plan
 *    _f32_mma_reason(arch, dtype, m, frag)  rocke_dconv_f32_mma_reason
 *    _zero_pad_operands(b, a, b_val, pad)   (static) dconv_zero_pad_operands
 *    _mma_f32_logical(b, arch, dtype, ...)  rocke_dconv_mma_f32_logical
 *    _mma_f32_chain(b, arch, dtype, ...)    rocke_dconv_mma_f32_chain
 *
 *  The 16-bit paths keep the exact ops they always emitted; every helper
 *  emits its ops in the Python order (byte identity).
 * ===================================================================== */

/* Widest single fp32 buffer / LDS access: one dwordx4 (Python _F32_IO_MAX). */
#define ROCKE_DCONV_F32_IO_MAX 4

const char* rocke_dconv_storage_dtype(const char* dtype)
{
    if(dtype == NULL)
        return NULL;
    if(strcmp(dtype, "fp16") == 0)
        return "fp16";
    if(strcmp(dtype, "bf16") == 0)
        return "bf16";
    if(strcmp(dtype, "fp32") == 0 || strcmp(dtype, "tf32") == 0)
        return "fp32";
    return NULL;
}

bool rocke_dconv_is_f32_storage(const char* dtype)
{
    const char* st = rocke_dconv_storage_dtype(dtype);
    return st != NULL && strcmp(st, "fp32") == 0;
}

int rocke_dconv_elem_bytes(const char* dtype)
{
    return rocke_dconv_is_f32_storage(dtype) ? 4 : 2;
}

const rocke_type_t* rocke_dconv_io_type(rocke_ir_builder_t* b, const char* dtype)
{
    const char* st = rocke_dconv_storage_dtype(dtype);
    if(st == NULL)
        return (const rocke_type_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "unsupported direct_conv dtype: %s%s%s; expected one of ['bf16', 'fp16', "
            "'fp32', 'tf32']",
            dtype ? "'" : "",
            dtype ? dtype : "None",
            dtype ? "'" : "");
    if(strcmp(st, "fp32") == 0)
        return rocke_f32();
    if(strcmp(st, "bf16") == 0)
        return rocke_bf16();
    return rocke_f16();
}

void rocke_dconv_mfma_dtype_flags(const char* dtype, int* f32, int* tf32)
{
    *f32 = (dtype != NULL && strcmp(dtype, "fp32") == 0) ? 1 : 0;
    *tf32 = (dtype != NULL && strcmp(dtype, "tf32") == 0) ? 1 : 0;
}

/* Pairwise vec_concat until one value remains (Python's `while len(parts) > 1`). */
static rocke_value_t* dconv_concat_tree(rocke_ir_builder_t* b, rocke_value_t** parts, int n)
{
    while(n > 1)
    {
        int m = 0;
        for(int i = 0; i < n; i += 2)
            parts[m++] = rocke_b_vec_concat(b, parts[i], parts[i + 1]);
        n = m;
    }
    return parts[0];
}

/* vec_pack of the ROCKE_DCONV_F32_IO_MAX elements of `val` from `i`. */
static rocke_value_t* dconv_f32_chunk(rocke_ir_builder_t* b, rocke_value_t* val, int i)
{
    rocke_value_t* elems[ROCKE_DCONV_F32_IO_MAX];
    for(int j = 0; j < ROCKE_DCONV_F32_IO_MAX; ++j)
        elems[j] = rocke_b_vec_extract(b, val, i + j);
    return rocke_b_vec_pack(b, elems, ROCKE_DCONV_F32_IO_MAX, rocke_f32());
}

/* `base` for chunk 0, else base + const_i32(step * i) -- Python's
 * `voff if i == 0 else b.add(voff, b.const_i32(step * i))`. */
static rocke_value_t* dconv_chunk_off(rocke_ir_builder_t* b, rocke_value_t* base, int i, int step)
{
    if(i == 0)
        return base;
    rocke_value_t* c = rocke_b_const_i32(b, step * i);
    return rocke_b_add(b, base, c);
}

rocke_value_t* rocke_dconv_load_elems(rocke_ir_builder_t* b,
                                      const char* dtype,
                                      rocke_value_t* rsrc,
                                      rocke_value_t* voff,
                                      rocke_value_t* soff,
                                      int n)
{
    if(!rocke_dconv_is_f32_storage(dtype))
    {
        if(n % 2)
            return (rocke_value_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "16-bit direct-conv load of %d elements is not dwords", n);
        return rocke_dconv_buf_load_vN(
            b, dtype != NULL && strcmp(dtype, "bf16") == 0, rsrc, voff, soff, n / 2);
    }
    if(n <= ROCKE_DCONV_F32_IO_MAX)
        return rocke_b_buffer_load_vN(b, rsrc, voff, soff, rocke_f32(), n);
    if(n % ROCKE_DCONV_F32_IO_MAX)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "fp32 direct-conv load of %d elements: not a multiple of 4", n);
    rocke_value_t* parts[64];
    int np = 0;
    for(int i = 0; i < n; i += ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_value_t* off = dconv_chunk_off(b, voff, i, 4);
        parts[np++]
            = rocke_b_buffer_load_vN(b, rsrc, off, soff, rocke_f32(), ROCKE_DCONV_F32_IO_MAX);
    }
    return dconv_concat_tree(b, parts, np);
}

void rocke_dconv_store_elems(rocke_ir_builder_t* b,
                             const char* dtype,
                             rocke_value_t* rsrc,
                             rocke_value_t* voff,
                             rocke_value_t* soff,
                             rocke_value_t* val,
                             int n)
{
    if(!rocke_dconv_is_f32_storage(dtype))
    {
        if(n % 2)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "16-bit direct-conv store of %d elements is not dwords", n);
            return;
        }
        rocke_dconv_buf_store_vN(
            b, dtype != NULL && strcmp(dtype, "bf16") == 0, rsrc, voff, soff, val, n / 2);
        return;
    }
    if(n <= ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_b_buffer_store_vN_f32(b, rsrc, voff, soff, val, n);
        return;
    }
    if(n % ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "fp32 direct-conv store of %d elements: not a multiple of 4", n);
        return;
    }
    for(int i = 0; i < n; i += ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_value_t* chunk = dconv_f32_chunk(b, val, i);
        rocke_value_t* off = dconv_chunk_off(b, voff, i, 4);
        rocke_b_buffer_store_vN_f32(b, rsrc, off, soff, chunk, ROCKE_DCONV_F32_IO_MAX);
    }
}

void rocke_dconv_store_from_f32(rocke_ir_builder_t* b,
                                const char* dtype,
                                rocke_value_t* rsrc,
                                rocke_value_t* voff,
                                rocke_value_t* soff,
                                rocke_value_t* acc,
                                int n)
{
    if(!rocke_dconv_is_f32_storage(dtype))
    {
        const bool bf16 = dtype != NULL && strcmp(dtype, "bf16") == 0;
        rocke_value_t* h = rocke_dconv_trunc_f32(b, bf16, acc);
        rocke_dconv_buf_store_vN(b, bf16, rsrc, voff, soff, h, n / 2);
        return;
    }
    rocke_dconv_store_elems(b, dtype, rsrc, voff, soff, acc, n);
}

rocke_value_t* rocke_dconv_smem_load_elems(rocke_ir_builder_t* b,
                                           const char* dtype,
                                           rocke_value_t* lds,
                                           rocke_value_t* row,
                                           rocke_value_t* idx,
                                           int n)
{
    const rocke_type_t* io_type = rocke_dconv_io_type(b, dtype);
    if(io_type == NULL)
        return NULL;
    if(!rocke_dconv_is_f32_storage(dtype) || n <= ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_value_t* ind[2] = {row, idx};
        return rocke_b_smem_load_vN(b, lds, ind, 2, io_type, n);
    }
    if(n % ROCKE_DCONV_F32_IO_MAX)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "fp32 direct-conv LDS read of %d elements: not a multiple of 4", n);
    rocke_value_t* parts[64];
    int np = 0;
    for(int i = 0; i < n; i += ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_value_t* ind[2] = {row, dconv_chunk_off(b, idx, i, 1)};
        parts[np++] = rocke_b_smem_load_vN(b, lds, ind, 2, rocke_f32(), ROCKE_DCONV_F32_IO_MAX);
    }
    return dconv_concat_tree(b, parts, np);
}

void rocke_dconv_smem_store_elems(rocke_ir_builder_t* b,
                                  const char* dtype,
                                  rocke_value_t* lds,
                                  rocke_value_t* row,
                                  rocke_value_t* idx,
                                  rocke_value_t* val,
                                  int n)
{
    if(!rocke_dconv_is_f32_storage(dtype) || n <= ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_value_t* ind[2] = {row, idx};
        rocke_b_smem_store_vN(b, lds, ind, 2, val, n);
        return;
    }
    if(n % ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "fp32 direct-conv LDS write of %d elements: not a multiple of 4",
                        n);
        return;
    }
    for(int i = 0; i < n; i += ROCKE_DCONV_F32_IO_MAX)
    {
        rocke_value_t* chunk = dconv_f32_chunk(b, val, i);
        rocke_value_t* ind[2] = {row, dconv_chunk_off(b, idx, i, 1)};
        rocke_b_smem_store_vN(b, lds, ind, 2, chunk, ROCKE_DCONV_F32_IO_MAX);
    }
}

/* Python _f32_mma_ops: the catalog's m x m MMA ops serving fp32-storage
 * `dtype` (tf32 -> the TF32 atoms, fp32 -> the exact fp32 ones), stably
 * sorted by fragment length, narrowest first. Returns the count. */
static int dconv_f32_mma_ops(
    const char* arch, const char* dtype, int m, const rocke_mma_op_t** out, int cap)
{
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch);
    if(t == NULL)
        return 0;
    const char* mma_dt = (dtype != NULL && strcmp(dtype, "tf32") == 0) ? "tf32" : "fp32";
    int n_ops = 0;
    const rocke_mma_op_t* ops = rocke_mma_catalog_ops(&t->mma, &n_ops);
    int n = 0;
    for(int i = 0; i < n_ops && n < cap; ++i)
    {
        const rocke_mma_op_t* op = &ops[i];
        if(op->family == NULL || strcmp(op->family, "mma") != 0)
            continue;
        if(op->srcs[0].dtype == NULL || strcmp(op->srcs[0].dtype, mma_dt) != 0)
            continue;
        if(op->srcs[1].dtype == NULL || strcmp(op->srcs[1].dtype, mma_dt) != 0)
            continue;
        if(op->srcs[2].dtype == NULL || strcmp(op->srcs[2].dtype, "fp32") != 0)
            continue;
        if(op->m != m || op->n != m)
            continue;
        /* stable insertion by a_frag_len */
        int j = n;
        while(j > 0 && out[j - 1]->srcs[0].frag_len > op->srcs[0].frag_len)
        {
            out[j] = out[j - 1];
            --j;
        }
        out[j] = op;
        ++n;
    }
    return n;
}

bool rocke_dconv_f32_mma_plan(
    const char* arch, const char* dtype, int m, int frag, rocke_dconv_f32_mma_plan_t* plan)
{
    const rocke_mma_op_t* ops[16];
    int n = dconv_f32_mma_ops(arch, dtype, m, ops, 16);
    for(int i = n - 1; i >= 0; --i)
    {
        const int len = ops[i]->srcs[0].frag_len;
        if(len > 0 && frag % len == 0)
        {
            plan->op = ops[i];
            plan->chunks = frag / len;
            plan->pad = 0;
            return true;
        }
    }
    for(int i = 0; i < n; ++i)
    {
        const int len = ops[i]->srcs[0].frag_len;
        if(len > frag)
        {
            plan->op = ops[i];
            plan->chunks = 1;
            plan->pad = len - frag;
            return true;
        }
    }
    return false;
}

bool rocke_dconv_f32_mma_reason(
    const char* arch, const char* dtype, int m, int frag, char* why, size_t why_cap)
{
    rocke_dconv_f32_mma_plan_t plan;
    if(rocke_dconv_f32_mma_plan(arch, dtype, m, frag, &plan))
        return false;
    if(why != NULL && why_cap > 0)
        snprintf(why,
                 why_cap,
                 "no %dx%d %s MFMA atom on %s",
                 m,
                 m,
                 (dtype != NULL && strcmp(dtype, "tf32") == 0) ? "TF32" : "fp32",
                 arch);
    return true;
}

/* Elements per lane of an operand fragment (a scalar counts as one). */
static int dconv_frag_width(const rocke_value_t* v)
{
    return v->type->kind == ROCKE_TYPE_VECTOR ? v->type->count : 1;
}

/* Python _zero_pad_operands: both operands as <frag + pad x float>, zero-filled
 * at the same K. */
static bool dconv_zero_pad_operands(rocke_ir_builder_t* b,
                                    rocke_value_t** a,
                                    rocke_value_t** b_val,
                                    int pad)
{
    const int frag = dconv_frag_width(*a);
    if(dconv_frag_width(*b_val) != frag)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "MMA operand fragments differ in width: A %d, B %d",
                        frag,
                        dconv_frag_width(*b_val));
        return false;
    }
    if(frag == 1)
    {
        *a = rocke_b_vec_pack(b, a, 1, rocke_f32());
        *b_val = rocke_b_vec_pack(b, b_val, 1, rocke_f32());
    }
    if(pad)
    {
        rocke_value_t* elems[64];
        rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
        for(int i = 0; i < frag; ++i)
            elems[i] = rocke_b_vec_extract(b, *a, i);
        for(int i = 0; i < pad; ++i)
            elems[frag + i] = zero;
        *a = rocke_b_vec_pack(b, elems, frag + pad, rocke_f32());
        for(int i = 0; i < frag; ++i)
            elems[i] = rocke_b_vec_extract(b, *b_val, i);
        for(int i = 0; i < pad; ++i)
            elems[frag + i] = zero;
        *b_val = rocke_b_vec_pack(b, elems, frag + pad, rocke_f32());
    }
    return true;
}

/* Python _mma_f32_logical's _piece. */
static rocke_value_t* dconv_mma_piece(
    rocke_ir_builder_t* b, const char* dtype, rocke_value_t* v, int i, int n, int chunks)
{
    const bool tf32 = dtype != NULL && strcmp(dtype, "tf32") == 0;
    rocke_value_t* piece;
    if(n == 1 && !tf32)
        return rocke_b_vec_extract(b, v, i);
    if(chunks == 1)
        piece = v;
    else
    {
        rocke_value_t* elems[64];
        for(int j = 0; j < n; ++j)
            elems[j] = rocke_b_vec_extract(b, v, i * n + j);
        piece = rocke_b_vec_pack(b, elems, n, rocke_f32());
    }
    if(tf32)
        return rocke_b_bitcast(b, piece, rocke_vector_type(b, rocke_tf32(), n));
    return piece;
}

rocke_value_t* rocke_dconv_mma_f32_logical(rocke_ir_builder_t* b,
                                           const char* arch,
                                           const char* dtype,
                                           int m,
                                           rocke_value_t* a,
                                           rocke_value_t* b_val,
                                           rocke_value_t* acc)
{
    if(!rocke_ir_builder_ok(b))
        return NULL;
    const int frag = dconv_frag_width(a);
    rocke_dconv_f32_mma_plan_t plan;
    if(!rocke_dconv_f32_mma_plan(arch, dtype, m, frag, &plan))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "no %dx%d %s MMA atom on %s", m, m, dtype, arch);
    if(!dconv_zero_pad_operands(b, &a, &b_val, plan.pad))
        return NULL;
    const int n = plan.op->srcs[0].frag_len;
    for(int i = 0; i < plan.chunks; ++i)
    {
        rocke_value_t* pa = dconv_mma_piece(b, dtype, a, i, n, plan.chunks);
        rocke_value_t* pb = dconv_mma_piece(b, dtype, b_val, i, n, plan.chunks);
        acc = rocke_b_mma(b, plan.op->op_id, pa, pb, acc, NULL, 0);
    }
    return acc;
}

rocke_value_t* rocke_dconv_mma_f32_chain(rocke_ir_builder_t* b,
                                         const char* arch,
                                         const char* dtype,
                                         int m,
                                         rocke_value_t* const* as,
                                         rocke_value_t* const* bs,
                                         int n,
                                         rocke_value_t* acc)
{
    if(n == 0 || !rocke_ir_builder_ok(b))
        return acc;
    const int frag = dconv_frag_width(as[0]);
    rocke_dconv_f32_mma_plan_t plan, wide;
    const bool has_plan = rocke_dconv_f32_mma_plan(arch, dtype, m, frag, &plan);
    const bool has_wide = rocke_dconv_f32_mma_plan(arch, dtype, m, 2 * frag, &wide);
    const bool fuse = has_plan && plan.pad > 0 && has_wide && wide.pad == 0;
    int i = 0;
    while(i < n)
    {
        rocke_value_t* a;
        rocke_value_t* bv;
        if(fuse && i + 1 < n)
        {
            a = rocke_b_vec_concat(b, as[i], as[i + 1]);
            bv = rocke_b_vec_concat(b, bs[i], bs[i + 1]);
            i += 2;
        }
        else
        {
            a = as[i];
            bv = bs[i];
            i += 1;
        }
        acc = rocke_dconv_mma_f32_logical(b, arch, dtype, m, a, bv, acc);
    }
    return acc;
}

/* ===================================================================== *
 *  Scalar-FMA kernels (scalar dgrad, the depthwise family): element I/O.
 *
 *    Python (conv_direct_grouped.py)        C++ (here)
 *    ------------------------------------   ---------------------------------
 *    _scalar_dtype_reason(kernel, dtype)    rocke_dconv_scalar_dtype_reason
 *    _scalar_name_flags(dtype)["f32"]       rocke_dconv_scalar_f32_flag
 *    _scalar_load(b, dtype, ...)            rocke_dconv_scalar_load
 *    _scalar_store(b, dtype, ...)           rocke_dconv_scalar_store
 *
 *  These kernels widen every element to f32 on load and narrow once on store,
 *  so fp32 storage only changes the load / store form and the byte stride;
 *  "tf32" runs the same exact fp32 math (one binary, the "f32" name flag).
 * ===================================================================== */

bool rocke_dconv_scalar_dtype_reason(const char* kernel,
                                     const char* dtype,
                                     char* why,
                                     size_t why_cap)
{
    if(dtype != NULL
       && (strcmp(dtype, "fp16") == 0 || strcmp(dtype, "bf16") == 0 || strcmp(dtype, "fp32") == 0
           || strcmp(dtype, "tf32") == 0))
        return false;
    if(why != NULL && why_cap > 0)
        snprintf(why,
                 why_cap,
                 "%s: unsupported dtype %s%s%s; expected fp16, bf16, fp32 or tf32 (fp32 and tf32 "
                 "both run exact fp32 math)",
                 kernel,
                 dtype ? "'" : "",
                 dtype ? dtype : "None",
                 dtype ? "'" : "");
    return true;
}

int rocke_dconv_scalar_f32_flag(const char* dtype)
{
    return (dtype != NULL && (strcmp(dtype, "fp32") == 0 || strcmp(dtype, "tf32") == 0)) ? 1 : 0;
}

rocke_value_t* rocke_dconv_scalar_load(rocke_ir_builder_t* b,
                                       const char* dtype,
                                       rocke_value_t* rsrc,
                                       rocke_value_t* voff,
                                       rocke_value_t* soff)
{
    if(dtype != NULL && strcmp(dtype, "bf16") == 0)
        return rocke_b_buffer_load_bf16(b, rsrc, voff, soff);
    if(rocke_dconv_scalar_f32_flag(dtype))
    {
        /* The one-dword buffer_load_vN (an op both engines lower) rather than
         * the scalar buffer_load. */
        rocke_value_t* v = rocke_b_buffer_load_vN(b, rsrc, voff, soff, rocke_f32(), 1);
        return rocke_b_vec_extract(b, v, 0);
    }
    return rocke_b_buffer_load_f16(b, rsrc, voff, soff);
}

void rocke_dconv_scalar_store(rocke_ir_builder_t* b,
                              const char* dtype,
                              rocke_value_t* rsrc,
                              rocke_value_t* voff,
                              rocke_value_t* soff,
                              rocke_value_t* acc)
{
    if(dtype != NULL && strcmp(dtype, "bf16") == 0)
    {
        rocke_value_t* h = rocke_b_trunc_f32_to_bf16(b, acc);
        rocke_b_buffer_store_bf16(b, rsrc, voff, soff, h);
    }
    else if(rocke_dconv_scalar_f32_flag(dtype))
        rocke_b_buffer_store_f32(b, rsrc, voff, soff, acc);
    else
    {
        rocke_value_t* h = rocke_b_trunc_f32_to_f16(b, acc);
        rocke_b_buffer_store_f16(b, rsrc, voff, soff, h);
    }
}

const rocke_type_t* rocke_dconv_dw_col_storage(const char* dtype)
{
    /* `dtype is TF32 or str(dtype).lower() == "tf32"` -> F32 */
    if(dtype != NULL && strlen(dtype) == 4 && tolower((unsigned char)dtype[0]) == 't'
       && tolower((unsigned char)dtype[1]) == 'f' && dtype[2] == '3' && dtype[3] == '2')
        return rocke_f32();
    return rocke_fuse_dtype_to_ir_str(dtype);
}
