// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_fwd_wmma_gate.cpp -- C++ port of library/kernels/common/
 * conv_fwd_wmma_gate.py: the capability gate the wave32 WMMA forward
 * convolutions share (direct, small C, small K).
 */
#include "rocke/instance_conv_fwd_wmma_gate_internal.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason, rocke_mma_catalog_has_shape */

namespace
{

void gate_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", msg);
    }
}

const char* gate_dtype(const rocke_direct_conv_problem_t* p)
{
    return p->dtype ? p->dtype : "fp16";
}

} // namespace

const char* rocke_wmma_ab_dtype(const char* dtype)
{
    return dtype && strcmp(dtype, "bf16") == 0 ? "bf16" : "f16";
}

bool rocke_wmma_fwd_gate(const rocke_direct_conv_problem_t* p,
                         const char* arch,
                         const char* name,
                         int wave_size,
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
        gate_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!target->memory.has_tdm)
    {
        snprintf(msg, sizeof msg, "no TDM on %s", arch);
        gate_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(strcmp(gate_dtype(p), "fp16") != 0 && strcmp(gate_dtype(p), "bf16") != 0)
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", gate_dtype(p));
        gate_set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* ab = rocke_wmma_ab_dtype(gate_dtype(p));
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
        gate_set_reason(reason, reason_cap, msg);
        return false;
    }
    return true;
}
