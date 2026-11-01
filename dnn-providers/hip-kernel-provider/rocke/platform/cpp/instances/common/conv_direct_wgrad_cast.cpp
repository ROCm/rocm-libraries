// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_wgrad_cast.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_wgrad_cast.py: a flat elementwise
 * narrowing cast, D[i] = cast_to_dtype_d(A[i]) for i in [0, count).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_wgrad_cast.h"

#include <string.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_wgrad_internal.hpp"

using namespace rocke_dwg;

namespace
{

bool is_bf16(const rocke_direct_wgrad_cast_spec_t* s)
{
    return s->dtype_d != NULL && strcmp(s->dtype_d, "bf16") == 0;
}

bool is_fp16(const rocke_direct_wgrad_cast_spec_t* s)
{
    return s->dtype_d != NULL && strcmp(s->dtype_d, "fp16") == 0;
}

} // namespace

extern "C" rocke_direct_wgrad_cast_spec_t rocke_direct_wgrad_cast_spec_default(void)
{
    rocke_direct_wgrad_cast_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = "direct_wgrad_cast";
    s.threads = 256;
    return s;
}

extern "C" bool rocke_direct_wgrad_cast_is_valid_spec(const rocke_direct_wgrad_cast_spec_t* s,
                                                      char* reason,
                                                      size_t reason_cap)
{
    if(!is_bf16(s) && !is_fp16(s))
    {
        snprintf(reason, reason_cap, "unsupported dtype_d %s; expected 'fp16' or 'bf16'",
                 s->dtype_d ? s->dtype_d : "(null)");
        return false;
    }
    if(s->threads < 1 || s->threads > 1024)
    {
        snprintf(reason, reason_cap, "threads must be in [1, 1024]");
        return false;
    }
    return true;
}

extern "C" rocke_status_t rocke_direct_wgrad_cast_kernel_name(
    const rocke_direct_wgrad_cast_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    char t_buf[24];
    snprintf(t_buf, sizeof t_buf, "t%d", s->threads);
    const char* parts[2] = {s->dtype_d, t_buf};
    return rocke_kernel_name_join(s->name, parts, 2, NULL, NULL, 0, out, out_cap, NULL);
}

extern "C" void rocke_direct_wgrad_cast_launch_geometry(const rocke_direct_wgrad_cast_spec_t* s,
                                                        int count,
                                                        int grid[3],
                                                        int block[3])
{
    grid[0] = cdiv(count, s->threads);
    grid[1] = 1;
    grid[2] = 1;
    block[0] = s->threads;
    block[1] = 1;
    block[2] = 1;
}

namespace
{

rocke_kernel_def_t* build_cast(rocke_ir_builder_t* b, const rocke_direct_wgrad_cast_spec_t* spec)
{
    Em e{b};
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->threads);
    rocke_param_opts_t ro;
    memset(&ro, 0, sizeof(ro));
    ro.noalias = ro.noalias_set = true;
    ro.readonly = ro.readonly_set = true;
    ro.align = 4;
    ro.align_set = true;
    const bool bf16 = is_bf16(spec);
    const rocke_type_t* dw_type = bf16 ? rocke_bf16() : rocke_f16();
    const int dw_align = 2;
    rocke_param_opts_t wo;
    memset(&wo, 0, sizeof(wo));
    wo.noalias = wo.noalias_set = true;
    wo.writeonly = wo.writeonly_set = true;
    wo.align = dw_align;
    wo.align_set = true;
    V a = rocke_b_param(b, "A", rocke_ptr_type(b, rocke_f32(), "global"), &ro);
    V dw = rocke_b_param(b, "D", rocke_ptr_type(b, dw_type, "global"), &wo);
    rocke_b_param(b, "A_bytes", rocke_i32(), NULL);
    rocke_b_param(b, "D_bytes", rocke_i32(), NULL);
    V count = rocke_b_param(b, "count", rocke_i32(), NULL);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    V bx = rocke_b_block_id_x(b);
    V bx_times_threads = e.muli(bx, spec->threads);
    V tid = rocke_b_thread_id_x(b);
    V i = e.add(bx_times_threads, tid);
    V ok = e.lt(i, count);
    V i_safe = e.select(ok, i, e.ci(0));
    V v_f32 = rocke_b_global_load_f32(b, a, i_safe, 4);
    V cast = bf16 ? rocke_b_trunc_f32_to_bf16(b, v_f32) : rocke_b_trunc_f32_to_f16(b, v_f32);
    e.if_(ok, [&]() { rocke_b_global_store(b, dw, i, cast, dw_align); });
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_direct_wgrad_cast(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_cast_spec_t* spec, const char* arch)
{
    (void)arch;
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_direct_wgrad_cast_is_valid_spec(spec, reason, sizeof reason))
    {
        fail_invalid(b, "invalid DirectWgradCastSpec", reason);
        return NULL;
    }
    return build_cast(b, spec);
}

extern "C" rocke_kernel_def_t* rocke_build_direct_wgrad_cast_new(
    rocke_ir_builder_t* b, const rocke_direct_wgrad_cast_spec_t* spec, const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_direct_wgrad_cast",
        [&](char* out, size_t cap) { return rocke_direct_wgrad_cast_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_direct_wgrad_cast(b, spec, arch); });
}

extern "C" rocke_status_t rocke_direct_wgrad_cast_lower_to_llvm(
    const rocke_direct_wgrad_cast_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    return lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_direct_wgrad_cast failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_direct_wgrad_cast_new(b, spec, arch); });
}
