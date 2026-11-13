// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_fwd_wmma_gate.cpp -- C++ port of library/kernels/common/
 * conv_fwd_wmma_gate.py: the capability gate, validator tail and halo
 * geometry of the wave32 WMMA forward convolutions (the users of each are
 * listed in rocke/instance_conv_fwd_wmma_gate_internal.h).
 */
#include "rocke/instance_conv_fwd_wmma_gate_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason, rocke_mma_catalog_has_shape */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_floor_div, rocke_ceil_div, set_reason */
#include "rocke/ir.h" /* rocke_scheduler_strategy_reason */
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */

const char* rocke_wmma_ab_dtype(const char* dtype)
{
    return dtype && strcmp(dtype, "bf16") == 0 ? "bf16" : "f16";
}

const char* rocke_wmma_fwd_op_id(rocke_ir_builder_t* b,
                                 const char* dtype,
                                 const char* arch,
                                 const char* who)
{
    const char* ab = rocke_wmma_ab_dtype(dtype);
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    const rocke_mmaop_t* op = rocke_archtarget_op_for_shape(
        target, "wmma", ab, ab, "fp32", ROCKE_FWD_WMMA_M, ROCKE_FWD_WMMA_N, ROCKE_FWD_WMMA_K, NULL);
    if(op == NULL || op->op_id == NULL)
    {
        const std::string msg = std::string(who) + ": no WMMA atom";
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    return op->op_id;
}

bool rocke_wmma_fwd_gate(const rocke_direct_conv_problem_t* p,
                         const char* arch,
                         const char* name,
                         int wave_size,
                         bool per_axis_pad,
                         const rocke_archtarget_t** target_out,
                         char* reason,
                         size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target_out)
        *target_out = target;
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(target->wave_size != 32 || wave_size != 32)
    {
        snprintf(msg, sizeof msg, "%s is a wave32 kernel", name);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!target->memory.has_tdm)
    {
        snprintf(msg, sizeof msg, "no TDM on %s", arch);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* dtype = rocke_direct_conv_problem_dtype(p);
    if(strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0)
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", dtype);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* ab = rocke_wmma_ab_dtype(dtype);
    if(!rocke_mma_catalog_has_shape(rocke_archtarget_mma(target),
                                    "wmma",
                                    ab,
                                    ab,
                                    "fp32",
                                    ROCKE_FWD_WMMA_M,
                                    ROCKE_FWD_WMMA_N,
                                    ROCKE_FWD_WMMA_K))
    {
        snprintf(msg,
                 sizeof msg,
                 "missing wmma_f32_%dx%dx%d_%s on %s",
                 ROCKE_FWD_WMMA_M,
                 ROCKE_FWD_WMMA_N,
                 ROCKE_FWD_WMMA_K,
                 ab,
                 arch);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(rocke_direct_problem_dims_reason(p, reason, reason_cap))
        return false;
    if(per_axis_pad)
        return !rocke_direct_conv_pad_w_reason(p, "", reason, reason_cap);
    return !rocke_direct_conv_one_padding_reason(p, "", reason, reason_cap);
}

bool rocke_wmma_fwd_limits_ok(const rocke_direct_conv_problem_t* p,
                              const char* sched,
                              long long n_blocks,
                              char* reason,
                              size_t reason_cap)
{
    char msg[512];
    if(rocke_scheduler_strategy_reason(sched, reason, reason_cap))
        return false;
    if(!rocke_wmma_fwd_tensor_bytes_ok(p, reason, reason_cap))
        return false;
    if(n_blocks > INT32_MAX)
    {
        snprintf(msg, sizeof msg, "%lld workgroups exceed the i32 block id", n_blocks);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    return true;
}

bool rocke_wmma_fwd_tensor_bytes_ok(const rocke_direct_conv_problem_t* p,
                                    char* reason,
                                    size_t reason_cap)
{
    char msg[512];
    const long long total_c = rocke_direct_conv_problem_total_c(p);
    const long long total_k = rocke_direct_conv_problem_total_k(p);
    const long long Ho = rocke_direct_conv_problem_Ho(p);
    const long long Wo = rocke_direct_conv_problem_Wo(p);
    const long long bytes[3] = {2LL * p->N * p->H * p->W * total_c,
                                2LL * total_k * p->KH * p->KW * p->cpg,
                                2LL * p->N * Ho * Wo * total_k};
    long long mx = bytes[0];
    if(bytes[1] > mx)
        mx = bytes[1];
    if(bytes[2] > mx)
        mx = bytes[2];
    if(mx > INT32_MAX)
    {
        snprintf(msg, sizeof msg, "tensors of %lld B exceed the i32 byte offsets", mx);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    return true;
}

int rocke_wmma_fwd_halo_extent(int stride, int tile, int k, int dil)
{
    return stride * (tile - 1) + (k - 1) * dil + 1;
}

int rocke_wmma_fwd_halo_cols(const rocke_direct_conv_problem_t* p, int tile_w, bool phase_split)
{
    if(phase_split)
        return tile_w + (int)rocke_floor_div((long long)(p->KW - 1) * p->dil_w, p->stride);
    return rocke_wmma_fwd_halo_extent(p->stride, tile_w, p->KW, p->dil_w);
}

int rocke_wmma_fwd_lead_cols(int pad_w, int stride, bool phase_split)
{
    return phase_split ? (int)rocke_ceil_div(pad_w, stride) : pad_w;
}
