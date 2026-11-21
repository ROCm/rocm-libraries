/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of library/kernels/common/conv_fwd_split_k_reduce.py
 *
 * Stage 2 of the split-K forward convolution: sum the ks_count fp32 slabs
 * Stage 1 stored and cast into D. Each thread owns `vec` consecutive output
 * elements: one vector load per slab, `vec` adds, one vector store. Every SSA
 * value is bound in the Python builder's evaluation order.
 */

#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_fwd_split_k_reduce.h"

#include "rocke/arena.h" /* rocke_arena_alloc */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.helpers.io.h" /* io_ir_type, pack_f32_to, store_scalar_from_f32 */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, SignatureBuilder */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

/* The widest per-thread vector: vec <= 4. */
#define ROCKE_FWD_SPLITK_REDUCE_MAX_VEC 4

/* ---- spec helpers -------------------------------------------------------- */

rocke_fwd_split_k_reduce_spec_t rocke_fwd_split_k_reduce_spec_default(void)
{
    rocke_fwd_split_k_reduce_spec_t s;
    memset(&s, 0, sizeof(s));
    s.dtype_d = "bf16";
    s.vec = 4;
    s.block_size = 256;
    s.degree = 0;
    s.name = "conv_fwd_splitk_reduce";
    return s;
}

rocke_status_t rocke_fwd_split_k_reduce_kernel_name(const rocke_fwd_split_k_reduce_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap)
{
    /* Python: kernel_name_join(name, f"v{vec}", f"b{block_size}",
     *                          [f"s{degree}"] if degree, dtype_d) */
    char v_buf[24];
    char b_buf[24];
    char s_buf[24];
    const char* parts[4];
    int n_parts = 0;
    if(spec == NULL || out == NULL || spec->dtype_d == NULL)
        return ROCKE_ERR_VALUE;
    snprintf(v_buf, sizeof(v_buf), "v%d", spec->vec);
    snprintf(b_buf, sizeof(b_buf), "b%d", spec->block_size);
    parts[n_parts++] = v_buf;
    parts[n_parts++] = b_buf;
    if(spec->degree)
    {
        snprintf(s_buf, sizeof(s_buf), "s%d", spec->degree);
        parts[n_parts++] = s_buf;
    }
    parts[n_parts++] = spec->dtype_d;
    return rocke_kernel_name_join(spec->name ? spec->name : "conv_fwd_splitk_reduce",
                                  parts,
                                  (size_t)n_parts,
                                  NULL,
                                  NULL,
                                  0,
                                  out,
                                  out_cap,
                                  NULL);
}

bool rocke_fwd_split_k_reduce_is_valid_spec(const rocke_fwd_split_k_reduce_spec_t* spec,
                                            char* reason,
                                            size_t reason_cap)
{
    const char* dtype_d;
    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
            snprintf(reason, reason_cap, "spec is NULL");
        return false;
    }
    if(spec->vec != 1 && spec->vec != 2 && spec->vec != 4)
    {
        if(reason && reason_cap > 0)
            snprintf(reason, reason_cap, "vec must be 1, 2 or 4 (got %d)", spec->vec);
        return false;
    }
    /* Python validate() rejects dtype_d=None; NULL is not a request for the
     * default (rocke_fwd_split_k_reduce_spec_default sets that). */
    dtype_d = spec->dtype_d;
    if(dtype_d == NULL)
    {
        if(reason && reason_cap > 0)
            snprintf(reason, reason_cap, "dtype_d must be fp16 or bf16 (got None)");
        return false;
    }
    if(strcmp(dtype_d, "fp16") != 0 && strcmp(dtype_d, "bf16") != 0)
    {
        if(reason && reason_cap > 0)
            snprintf(reason, reason_cap, "dtype_d must be fp16 or bf16 (got %s)", dtype_d);
        return false;
    }
    if(spec->degree < 0)
    {
        if(reason && reason_cap > 0)
            snprintf(reason, reason_cap, "degree must be >= 0 (got %d)", spec->degree);
        return false;
    }
    return true;
}

/* ---- main builder -------------------------------------------------------- */

/* Python _load(off): one f32 load (vec 1) or one vector load + extracts. */
static void fwd_reduce_load(
    rocke_ir_builder_t* b, rocke_value_t* ws_ptr, rocke_value_t* off, int vec, rocke_value_t** out)
{
    if(vec == 1)
    {
        out[0] = rocke_b_global_load_f32(b, ws_ptr, off, 4);
        return;
    }
    rocke_value_t* v = rocke_b_global_load_vN(b, ws_ptr, off, rocke_f32(), vec, 0);
    for(int i = 0; i < vec; ++i)
        out[i] = rocke_b_vec_extract(b, v, i);
}

rocke_kernel_def_t* rocke_build_conv_fwd_split_k_reduce(rocke_ir_builder_t* b,
                                                        const rocke_fwd_split_k_reduce_spec_t* spec,
                                                        const char* arch)
{
    char reason[256];
    (void)arch;
    if(b == NULL || spec == NULL)
        return NULL;
    if(!rocke_fwd_split_k_reduce_is_valid_spec(spec, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "fwd_split_k_reduce: %s", reason);
        return NULL;
    }

    const int vec = spec->vec;
    const char* dtype_d = spec->dtype_d; /* non-NULL: validated above */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    /* ---- params ---------------------------------------------------------- */
    rocke_param_opts_t ro;
    memset(&ro, 0, sizeof(ro));
    ro.noalias = ro.noalias_set = true;
    ro.readonly = ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;

    rocke_param_opts_t wo;
    memset(&wo, 0, sizeof(wo));
    wo.noalias = wo.noalias_set = true;
    wo.writeonly = wo.writeonly_set = true;
    wo.align = 16;
    wo.align_set = true;

    rocke_value_t* ws_ptr
        = rocke_b_param(b, "ws_ptr", rocke_ptr_type(b, rocke_f32(), "global"), &ro);
    rocke_value_t* d_ptr = rocke_b_param(
        b, "d_ptr", rocke_ptr_type(b, rocke_b_io_ir_type(b, dtype_d), "global"), &wo);
    rocke_value_t* elems = rocke_b_param(b, "elems", rocke_i32(), NULL);
    rocke_b_param(b, "ws_bytes", rocke_i32(), NULL); /* ABI boundary */
    rocke_b_param(b, "d_bytes", rocke_i32(), NULL); /* ABI boundary */
    rocke_value_t* ks_count = rocke_b_param(b, "ks_count", rocke_i32(), NULL);

    /* gid = block_id_x * block_size + thread_id_x; idx = gid * vec */
    rocke_value_t* bid = rocke_b_block_id_x(b);
    rocke_value_t* c_bs = rocke_b_const_i32(b, spec->block_size);
    rocke_value_t* base = rocke_b_mul(b, bid, c_bs);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* gid = rocke_b_add(b, base, tid);
    rocke_value_t* idx = rocke_b_mul(b, gid, rocke_b_const_i32(b, vec));

    rocke_value_t* total[ROCKE_FWD_SPLITK_REDUCE_MAX_VEC];
    rocke_if_t guard = rocke_b_scf_if(b, rocke_b_cmp_lt(b, idx, elems));
    rocke_b_region_enter(b, guard.then_region);
    if(spec->degree)
    {
        /* Unrolled: every slab's load is issued before the first add. */
        rocke_value_t** slabs = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, sizeof(rocke_value_t*) * (size_t)spec->degree * (size_t)vec);
        if(slabs == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "fwd_split_k_reduce: slab alloc failed");
            rocke_b_region_leave(b);
            return NULL;
        }
        for(int s = 0; s < spec->degree; ++s)
        {
            rocke_value_t* off = idx;
            if(s != 0)
            {
                rocke_value_t* c_s = rocke_b_const_i32(b, s);
                rocke_value_t* s_base = rocke_b_mul(b, c_s, elems);
                off = rocke_b_add(b, s_base, idx);
            }
            fwd_reduce_load(b, ws_ptr, off, vec, &slabs[(size_t)s * (size_t)vec]);
        }
        for(int i = 0; i < vec; ++i)
            total[i] = slabs[i];
        for(int s = 1; s < spec->degree; ++s)
        {
            for(int i = 0; i < vec; ++i)
                total[i] = rocke_b_fadd(b, total[i], slabs[(size_t)s * (size_t)vec + (size_t)i]);
        }
    }
    else
    {
        rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
        static const char* const kAccNames[ROCKE_FWD_SPLITK_REDUCE_MAX_VEC]
            = {"acc0", "acc1", "acc2", "acc3"};
        rocke_iter_arg_t init[ROCKE_FWD_SPLITK_REDUCE_MAX_VEC];
        for(int i = 0; i < vec; ++i)
        {
            init[i].name = kAccNames[i];
            init[i].init = zero;
        }
        rocke_value_t* c_lo = rocke_b_const_i32(b, 0);
        rocke_value_t* c_one = rocke_b_const_i32(b, 1);
        rocke_for_t loop = rocke_b_scf_for_iter(b,
                                                c_lo,
                                                ks_count,
                                                c_one,
                                                init,
                                                vec,
                                                "slab",
                                                /*unroll=*/false,
                                                /*elide_trailing_barrier=*/true);
        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* s_base = rocke_b_mul(b, loop.iv, elems);
            rocke_value_t* off = rocke_b_add(b, s_base, idx);
            rocke_value_t* part[ROCKE_FWD_SPLITK_REDUCE_MAX_VEC];
            rocke_value_t* sums[ROCKE_FWD_SPLITK_REDUCE_MAX_VEC];
            fwd_reduce_load(b, ws_ptr, off, vec, part);
            for(int i = 0; i < vec; ++i)
                sums[i] = rocke_b_fadd(b, loop.iter_vars[i], part[i]);
            rocke_b_scf_yield(b, sums, vec);
        }
        rocke_b_region_leave(b);
        for(int i = 0; i < vec; ++i)
            total[i] = loop.op->results[i];
    }
    if(vec == 1)
    {
        rocke_b_store_scalar_from_f32(b, d_ptr, idx, total[0], dtype_d);
    }
    else
    {
        rocke_value_t* packed = rocke_b_pack_f32_to(b, total, vec, dtype_d);
        rocke_b_global_store_vN(b, d_ptr, idx, packed, vec, 0);
    }
    rocke_b_region_leave(b);

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_conv_fwd_split_k_reduce_new(
    rocke_ir_builder_t* b, const rocke_fwd_split_k_reduce_spec_t* spec, const char* arch)
{
    if(b != NULL)
    {
        /* Zero first, so every early return below leaves a builder the caller
         * can safely query and rocke_ir_builder_free(). */
        memset(b, 0, sizeof(*b));
    }
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        char reason[256];
        if(b == NULL)
            return NULL;
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_conv_fwd_split_k_reduce: null spec");
            return NULL;
        }
        /* Python: spec.validate() runs before IRBuilder(spec.kernel_name()). */
        if(!rocke_fwd_split_k_reduce_is_valid_spec(spec, reason, sizeof(reason)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "fwd_split_k_reduce: %s", reason);
            return NULL;
        }
        if(rocke_fwd_split_k_reduce_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_conv_fwd_split_k_reduce: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_conv_fwd_split_k_reduce(b, spec, arch);
    });
}

void rocke_fwd_split_k_reduce_grid(const rocke_fwd_split_k_reduce_spec_t* spec,
                                   long long elems,
                                   int* out_gx,
                                   int* out_gy,
                                   int* out_gz)
{
    const long long per_block = (long long)spec->vec * (long long)spec->block_size;
    long long gx = per_block > 0 ? (elems + per_block - 1) / per_block : 1;
    *out_gx = (int)(gx < 1 ? 1 : gx);
    *out_gy = 1;
    *out_gz = 1;
}

rocke_status_t rocke_fwd_split_k_reduce_signature(rocke_arena_t* arena,
                                                  const rocke_fwd_split_k_reduce_spec_t* spec,
                                                  const rocke_sig_entry_t** out_items,
                                                  size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;
    if(arena == NULL || spec == NULL || spec->dtype_d == NULL || out_items == NULL
       || out_count == NULL)
        return ROCKE_ERR_VALUE;
    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
        return st;
    rocke_signature_builder_ptr(&sb, "ws_ptr", "fp32", NULL);
    rocke_signature_builder_ptr(&sb, "d_ptr", spec->dtype_d, NULL);
    rocke_signature_builder_scalar(&sb, "elems", "i32");
    rocke_signature_builder_scalar(&sb, "ws_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "d_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "ks_count", "i32");
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

rocke_status_t rocke_fwd_split_k_reduce_lower_to_llvm(const rocke_fwd_split_k_reduce_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k;
    rocke_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
        return ROCKE_ERR_VALUE;
    if(arch == NULL)
        arch = "gfx950";

    k = rocke_build_conv_fwd_split_k_reduce_new(&b, spec, arch);
    if(k == NULL)
    {
        /* Capture status BEFORE free (free memsets struct to zero). */
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
                m = "build_conv_fwd_split_k_reduce failed";
            n = strlen(m);
            if(n >= err_cap)
                n = err_cap - 1;
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(k, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
