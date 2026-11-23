// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * serialized.cpp -- rocke_lower_serialized_ir_to_llvm: serialized ck.dsl.ir/v1
 * text in, AMDGPU LLVM IR or the engine's refusal out. It composes the shared
 * parse step (rocke/serialized_ir_internal.h) with rocke_lower_kernel_to_llvm_ex,
 * the LLVM twin of rocke_lower_serialized_ir_to_hip_text. The rocke_engine
 * binding and the ctypes transport rocke.core.native_engine both lower a
 * Python-built kernel through it, so the C++ engine has one serialized-IR LLVM
 * entry.
 */
#include "rocke/lower_llvm.h"

#include "rocke/ir.h"
#include "rocke/serialized_ir_internal.h"
#include "rocke/strbuf.h"

rocke_status_t rocke_lower_serialized_ir_to_llvm(const char* ir_text,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_text,
                                                 char** out_message)
{
    if(out_message)
    {
        *out_message = nullptr;
    }
    rocke_strbuf_t message;
    rocke_strbuf_init(&message, 0);
    if(!out_text)
    {
        ckc::serialized_ir_report(&message, "no output buffer");
        ckc::serialized_ir_take_message(&message, out_message);
        return ROCKE_ERR_VALUE;
    }
    *out_text = nullptr;
    rocke_ir_builder_t b = {};
    rocke_kernel_def_t* kernel = nullptr;
    rocke_status_t status = ckc::serialized_ir_parse(ir_text, &b, &kernel, &message);
    if(status == ROCKE_OK)
    {
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = '\0';
        /* An empty arch names no target, as NULL does: there is no default. */
        status = rocke_lower_kernel_to_llvm_ex(
            kernel, flavor, (arch && *arch) ? arch : nullptr, out_text, err, sizeof err);
        rocke_ir_builder_free(&b);
        if(status != ROCKE_OK)
        {
            ckc::serialized_ir_report(&message, err[0] ? err : "unknown lowering error");
        }
    }
    if(status == ROCKE_OK)
    {
        rocke_strbuf_free(&message);
    }
    else
    {
        ckc::serialized_ir_take_message(&message, out_message);
    }
    return status;
}
